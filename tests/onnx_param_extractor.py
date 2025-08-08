# export_onnx_testbench_fixed.py

import onnx
import onnxruntime
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict

# ==============================================================================
# CLASS: ONNX Parameter Extractor (No changes needed here)
# ==============================================================================

class ONNXParameterExtractor:
    """
    Parses a quantized ONNX model to extract all static parameters (weights, 
    biases, scales, zero-points) and graph metadata for all relevant operations.
    """
    def __init__(self, onnx_model_path):
        print(f"🔎 Initializing ONNX extractor for: {onnx_model_path}")
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        
        # Pre-computation for faster lookups
        self.initializers = {init.name: onnx.numpy_helper.to_array(init) for init in self.graph.initializer}
        self.node_by_output = {output: node for node in self.graph.node for output in node.output}
        self.consumers_by_input = defaultdict(list)
        for node in self.graph.node:
            for input_name in node.input:
                self.consumers_by_input[input_name].append(node)
        
        self.value_info = {vi.name: vi for vi in self.graph.value_info}
        for i in self.graph.input: self.value_info[i.name] = i
        for o in self.graph.output: self.value_info[o.name] = o
        for i in self.graph.initializer: self.value_info[i.name] = i
        
        print("✅ Extractor ready. Initializers, node maps, and value info created.")

    def get_initializer(self, name):
        return self.initializers.get(name)

    def find_consumers(self, tensor_name):
        return self.consumers_by_input.get(tensor_name, [])

    def _trace_forward_to_node(self, start_tensor_name, target_op_type):
        passthrough_ops = {'QuantizeLinear', 'DequantizeLinear', 'ReLU', 'Reshape', 'Transpose', 'Flatten', 'Clip'}
        worklist = [start_tensor_name]
        visited_tensors = {start_tensor_name}

        while worklist:
            current_tensor = worklist.pop(0)
            for consumer_node in self.find_consumers(current_tensor):
                if consumer_node.op_type == target_op_type:
                    return consumer_node, current_tensor
                if consumer_node.op_type in passthrough_ops:
                    for output_tensor in consumer_node.output:
                        if output_tensor not in visited_tensors:
                            visited_tensors.add(output_tensor)
                            worklist.append(output_tensor)
        return None, None

    def _trace_input_to_dequant(self, input_name):
        dq_node = self.node_by_output.get(input_name)
        if not (dq_node and dq_node.op_type == 'DequantizeLinear'): return None
        if len(dq_node.input) < 2: return None
            
        q_tensor_name, scale_name = dq_node.input[0], dq_node.input[1]
        q_value, scale_value = self.get_initializer(q_tensor_name), self.get_initializer(scale_name)
        if scale_value is None: return None

        zp_value = None
        if len(dq_node.input) > 2:
            zp_value = self.get_initializer(dq_node.input[2])
        else:
            q_tensor_info = self.value_info.get(q_tensor_name)
            zp_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(q_tensor_info.type.tensor_type.elem_type, np.uint8) if q_tensor_info else np.uint8
            zp_value = np.zeros_like(scale_value, dtype=zp_dtype)
                
        if zp_value is None: return None

        return {'q_tensor_name': q_tensor_name, 'q_value': q_value, 'scale_value': scale_value, 'zp_value': zp_value}

    def _trace_output_to_quant(self, tensor_name):
        q_node, _ = self._trace_forward_to_node(tensor_name, 'QuantizeLinear')
        if q_node:
            scale, zp = self.get_initializer(q_node.input[1]), self.get_initializer(q_node.input[2])
            if scale is not None and zp is not None:
                return {'output_q_tensor': q_node.output[0], 'output_scale': scale, 'output_zp': zp}
        return None

    def extract_all_parameters(self):
        extracted_params = {'Linear': [], 'MatMul_Activation': [], 'Softmax': [], 'ReLU': []}
        
        for node in self.graph.node:
            if node.op_type == 'MatMul':
                matmul_node = node
                input_a_params = self._trace_input_to_dequant(matmul_node.input[0])
                input_b_params = self._trace_input_to_dequant(matmul_node.input[1])
                if not (input_a_params and input_b_params): continue

                weight_params, activation_params = (input_a_params, input_b_params) if input_a_params.get('q_value') is not None else (input_b_params, input_a_params)
                
                is_linear_layer = False
                add_node, add_input_from_matmul = self._trace_forward_to_node(matmul_node.output[0], 'Add')

                if add_node:
                    bias_input_idx = 1 if add_node.input[0] == add_input_from_matmul else 0
                    bias_input_name = add_node.input[bias_input_idx]
                    bias_params = self._trace_input_to_dequant(bias_input_name)
                    
                    if bias_params and bias_params.get('q_value') is not None:
                        output_q_info = self._trace_output_to_quant(add_node.output[0])
                        if output_q_info:
                            extracted_params['Linear'].append({
                                'name': matmul_node.name, 'bias_add_node_name': add_node.name,
                                'activation': activation_params, 'weight': weight_params,
                                'bias': bias_params, **output_q_info,
                            })
                            is_linear_layer = True
                
                if not is_linear_layer:
                    output_q_info = self._trace_output_to_quant(matmul_node.output[0])
                    if output_q_info:
                        extracted_params['MatMul_Activation'].append({
                            'name': matmul_node.name, 'activation': activation_params,
                            'weights': weight_params, **output_q_info,
                        })

            elif node.op_type == 'Softmax' or node.op_type.lower() == 'relu':
                input_params = self._trace_input_to_dequant(node.input[0])
                output_q_info = self._trace_output_to_quant(node.output[0])
                if input_params and output_q_info:
                    op_key = 'Softmax' if node.op_type == 'Softmax' else 'ReLU'
                    extracted_params[op_key].append({'name': node.name, 'activation': input_params, **output_q_info})

        print(f"✅ Static Extraction Complete: Found {len(extracted_params['Linear'])} Linear Layers, "
              f"{len(extracted_params['MatMul_Activation'])} Activation MatMuls, "
              f"{len(extracted_params['Softmax'])} Softmax, and {len(extracted_params['ReLU'])} ReLU layers.")
        return extracted_params

# ==============================================================================
# MAIN SCRIPT and HELPERS (UPDATED PRINTING LOGIC)
# ==============================================================================

def get_ground_truth_input(image_path):
    if not os.path.exists(image_path): raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"🖼️  Using specific ground truth image: {image_path}")
    img = Image.open(image_path).convert('L')
    preprocess = transforms.Compose([transforms.Resize((60, 90)), transforms.ToTensor()])
    return preprocess(img).unsqueeze(0).numpy()

def run_and_capture_intermediates(onnx_path, input_feed, intermediate_tensor_names):
    print("🚀 Running inference to capture intermediate activations...")
    model = onnx.load(onnx_path)
    existing_outputs = {o.name for o in model.graph.output}
    for name in intermediate_tensor_names:
        if name not in existing_outputs:
             model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT8, None))
    
    session = onnxruntime.InferenceSession(model.SerializeToString())
    output_names = [out.name for out in session.get_outputs()]
    
    results = session.run(output_names, input_feed)
    results_map = dict(zip(output_names, results))
    print("✅ Intermediate values captured.")
    return results_map

def main(args):
    extractor = ONNXParameterExtractor(args.onnx_model)
    all_static_params = extractor.extract_all_parameters()
    
    input_image_np = get_ground_truth_input(args.image)
    input_feed = {
        "img_data": input_image_np,
        "additional_data": np.zeros((1, 1), dtype=np.float32), 
        "quat_data": np.zeros((1, 4), dtype=np.float32), 
        "h_in": np.zeros((3, 1, 128), dtype=np.float32), 
        "c_in": np.zeros((3, 1, 128), dtype=np.float32)
    }
    
    intermediate_names_to_capture = set()
    for op_type in all_static_params:
        for p in all_static_params[op_type]:
            if 'activation' in p and p['activation']:
                intermediate_names_to_capture.add(p['activation']['q_tensor_name'])
            if 'weights' in p and p['weights']:
                 intermediate_names_to_capture.add(p['weights']['q_tensor_name'])
            if 'output_q_tensor' in p and p['output_q_tensor']:
                 intermediate_names_to_capture.add(p['output_q_tensor'])
    
    all_runtime_values = run_and_capture_intermediates(args.onnx_model, input_feed, list(intermediate_names_to_capture))
    
    print("\n" + "="*80)
    print("         COMPLETE DATA FOR HARDWARE TESTBENCH GENERATION")
    print("="*80)

    def print_tensor_summary(label, tensor):
        if tensor is not None and tensor.size > 0: 
            min_val = f"{np.min(tensor):<8.4f}" if np.issubdtype(tensor.dtype, np.floating) else f"{np.min(tensor):<8}"
            max_val = f"{np.max(tensor):<8.4f}" if np.issubdtype(tensor.dtype, np.floating) else f"{np.max(tensor):<8}"
            print(f"  - {label:<25} Shape={str(tensor.shape):<20} DType={str(tensor.dtype):<10} Min={min_val} Max={max_val}")
        else: 
            print(f"  - {label:<25} Not Found or Empty")

    if all_static_params['Linear']:
        print(f"\n{'#'*15} LINEAR LAYERS (MatMul + Bias) {'#'*15}")
        for i, p in enumerate(sorted(all_static_params['Linear'], key=lambda x: x['name'])):
            print(f"\n--- Linear Layer {i+1}: {p['name']} --> {p['bias_add_node_name']} ---")
            print_tensor_summary('Input Activation (int8):', all_runtime_values.get(p['activation']['q_tensor_name']))
            print_tensor_summary('Weights (int8):', p['weight']['q_value'])
            print_tensor_summary('Bias (int32):', p['bias']['q_value'])
            print_tensor_summary('Output (int8):', all_runtime_values.get(p['output_q_tensor']))
            print("  - Quantization Parameters:")
            print(f"    - Activation: Scale={p['activation']['scale_value']}, ZP={p['activation']['zp_value']}")
            # <--- FIX: Changed printing from shape to value for clarity
            print(f"    - Weights:    Scale={p['weight']['scale_value']}, ZP={p['weight']['zp_value']}")
            print(f"    - Bias:       Scale={p['bias']['scale_value']}, ZP={p['bias']['zp_value']}")
            # --->
            print(f"    - Output:     Scale={p['output_scale']}, ZP={p['output_zp']}")

    for op_type in ['MatMul_Activation', 'ReLU', 'Softmax']:
        if all_static_params[op_type]:
            print(f"\n{'#'*15} {op_type.upper().replace('_', ' ')} LAYERS {'#'*15}")
            for i, p in enumerate(sorted(all_static_params[op_type], key=lambda x: x['name'])):
                print(f"\n--- {op_type} Layer {i+1}: {p['name']} ---")
                if op_type == 'MatMul_Activation':
                    print_tensor_summary('Input Activation 1 (int8):', all_runtime_values.get(p['activation']['q_tensor_name']))
                    print_tensor_summary('Input Activation 2 (int8):', all_runtime_values.get(p['weights']['q_tensor_name']))
                    print_tensor_summary('Output (int8):', all_runtime_values.get(p['output_q_tensor']))
                    print("  - Quantization Parameters:")
                    # <--- FIX: Changed printing from shape to value for clarity
                    print(f"    - Activation 1: Scale={p['activation']['scale_value']}, ZP={p['activation']['zp_value']}")
                    print(f"    - Activation 2: Scale={p['weights']['scale_value']}, ZP={p['weights']['zp_value']}")
                    # --->
                else:
                    print_tensor_summary('Input (int8):', all_runtime_values.get(p['activation']['q_tensor_name']))
                    print_tensor_summary('Output (int8):', all_runtime_values.get(p['output_q_tensor']))
                    print("  - Quantization Parameters:")
                    print(f"    - Input:        Scale={p['activation']['scale_value']}, ZP={p['activation']['zp_value']}")
                print(f"    - Output:       Scale={p['output_scale']}, ZP={p['output_zp']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract all static and dynamic parameters from an ONNX model for hardware verification.")
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to the final quantized ONNX model (.onnx)')
    parser.add_argument('--image', type=str, required=True, help='Path to the ground-truth input image.')
    args = parser.parse_args()
    main(args)