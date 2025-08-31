import torch
import torch.ao.quantization
import numpy as np
import argparse
import os
import sys
import re
from collections import defaultdict
from PIL import Image
from torchvision import transforms

# --- Project Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# --- Your Custom Imports ---
from models.ITA_single_layer_upsample_shuffle.QAT.model import ITALSTMNetVIT_QAT
#from models.ITA_layers_ReLU_QAT import HardwareReLU
from third_party.ITA_FPGA.PyITA.ITA import Transformer # Your accelerator simulator

# ==============================================================================
# 1. PYTORCH DATA EXTRACTOR (WITH CORRECTIONS)
# ==============================================================================

class PyTorchDataExtractor:
    def __init__(self, model):
        self.model = model
        self.captured_data = {}

    def _get_hook(self, name):
        def hook(module, input_tensors, output_tensor):
            self.captured_data[name]['input'] = []
            for i, in_tensor in enumerate(input_tensors):
                if in_tensor.is_quantized:
                    self.captured_data[name]['input'].append({
                        'q_value': in_tensor.int_repr().detach().clone().numpy(),
                        'scale': in_tensor.q_scale(),
                        'zp': in_tensor.q_zero_point()
                    })

            if output_tensor.is_quantized:
                self.captured_data[name]['output'] = {
                    'q_value': output_tensor.int_repr().detach().clone().numpy(),
                    'scale': output_tensor.q_scale(),
                    'zp': output_tensor.q_zero_point()
                }
        return hook

    def extract_and_capture(self, dummy_input):
        print("--- Step 1: Extracting PyTorch Parameters & Registering Hooks ---")
        
        for name, module in self.model.named_modules():
            # Condition 1: Handle Quantized Linear Layers
            if isinstance(module, torch.nn.quantized.Linear):
                self.captured_data[name] = {
                    'type': 'Linear',
                    'weight': {
                        'q_value': module.weight().int_repr().detach().clone().numpy(),
                        'scale': module.weight().q_scale(),
                        'zp': module.weight().q_zero_point()
                    },
                    'bias': module.bias().detach().clone().numpy() if module.bias() is not None else None,
                    'output_scale': module.scale,
                    'output_zp': module.zero_point
                }
                module.register_forward_hook(self._get_hook(name))

            # Condition 2: Handle Quantized ReLU Layers that were not fused
            elif isinstance(module, torch.nn.ReLU) and hasattr(module, 'scale'):
                print(f"INFO: Correctly capturing ReLU layer: {name}")
                self.captured_data[name] = {
                    'type': 'ReLU',
                    'output_scale': module.scale,
                    'output_zp': module.zero_point
                }
                module.register_forward_hook(self._get_hook(name))
                
            elif 'ffn_blocks' in name and isinstance(module, torch.nn.ReLU):
                print(f"INFO: Unconditionally capturing ReLU layer: {name}")
                # We create a placeholder entry. The hook will capture the I/O tensors.
                self.captured_data[name] = {'type': 'ReLU'}
                module.register_forward_hook(self._get_hook(name))
            # ...

            # Condition 3: Handle the Custom Softmax
            elif 'attention_blocks' in name and name.endswith('custom_softmax'):
                self.captured_data[name] = {'type': 'Softmax'}
                module.register_forward_hook(self._get_hook(name))

        print("✅ Static parameters extracted and hooks registered.")
        print("\n--- Step 2: Running Inference to Capture Ground Truth Activations ---")
        with torch.no_grad():
            self.model(dummy_input)
        print("✅ Intermediate values captured.")
        
        # Post-processing to fill in derived data
        for name, data in self.captured_data.items():
            if 'output' in data and 'output_scale' not in data:
                 data['output_scale'] = data['output']['scale']
                 data['output_zp'] = data['output']['zp']

        return self.captured_data

# ==============================================================================
# 2. HELPER FUNCTIONS (WITH CORRECTIONS)
# ==============================================================================

def calculate_hw_params(effective_scale: float, bits: int = 8) -> tuple[int, int]:
    """
    Converts a float scaling factor into an integer multiplier and a bit shift.
    """
    if effective_scale == 0:
        return 0, 0
    mantissa, exponent = np.frexp(effective_scale)
    multiplier = int(round(mantissa * (2**bits)))
    shift = bits - exponent
    return multiplier, shift


def custom_quantized_matmul(q_a: torch.Tensor,
                              q_b: torch.Tensor,
                              output_scale: float,
                              output_zero_point: int) -> torch.Tensor:
    """
    Performs quantized matrix multiplication that is robust to mixed
    qint8/quint8 inputs by manually handling the integer math. This version
    explicitly calculates the final integer values to align with hardware simulators.
    """
    # Dequantize inputs to their integer representations
    a_int = q_a.int_repr().to(torch.int32)
    b_int = q_b.int_repr().to(torch.int32)
    a_zp = q_a.q_zero_point()
    b_zp = q_b.q_zero_point()

    # Perform the matmul on zero-point-adjusted integers to get the int32 accumulator
    accumulator = torch.matmul(a_int - a_zp, b_int - b_zp)

    # <-- THIS IS THE CORRECT, EXPLICIT INTEGER MATH
    # Calculate the effective multiplier for requantization
    requantization_multiplier = (q_a.q_scale() * q_b.q_scale()) / output_scale
    
    # Apply the multiplier and add the output zero point
    output_unclamped = accumulator.float() * requantization_multiplier + output_zero_point
    
    # Clamp to the int8 range [-128, 127] and cast to the final integer type
    q_out_clamped = output_unclamped.round().clamp(-128, 127).to(torch.int8)

    # Package the result back into a quantized tensor
    return torch._make_per_tensor_quantized_tensor(
        q_out_clamped, scale=output_scale, zero_point=output_zero_point
    )

def organize_params_by_block(flat_params):
    organized = defaultdict(lambda: defaultdict(dict))
    
    # Check if this is a single-layer model (no index in module names)
    is_single_layer = not any('attention_blocks.0' in name or 'ffn_blocks.0' in name 
                              for name in flat_params.keys())
    
    if is_single_layer:
        # For single layer, assign everything to block index 0
        for name, params in flat_params.items():
            if 'attention_blocks' in name:
                layer_key = name.replace('attention_blocks.', '')
                
                # Handle special cases
                if layer_key == 'out_proj':
                    organized['attention'][0]['av_matmul'] = {
                        'type': 'MatMul_AV',
                        'input': None,
                        'output': params['input'][0]
                    }
                elif layer_key == 'custom_softmax':
                    layer_key = 'softmax'
                    organized['attention'][0]['qk_matmul'] = {
                        'type': 'MatMul_QK', 
                        'input': None, 
                        'output': params['input'][0]
                    }
                
                organized['attention'][0][layer_key] = params
                
            elif 'ffn_blocks' in name:
                layer_key = name.replace('ffn_blocks.', '')
                if layer_key == 'activation':
                    layer_key = 'relu'
                organized['ffn'][0][layer_key] = params
    else:
        # Original multi-layer logic
        pattern = re.compile(r"(?P<type>attention_blocks|ffn_blocks)\.(?P<index>\d+)\.(?P<layer>[\w_]+)")
        
        for name, params in flat_params.items():
            match = pattern.match(name)
            if match:
                d = match.groupdict()
                block_type = 'attention' if 'attention' in d['type'] else 'ffn'
                layer_key = d['layer']
                
                # ... rest of your original logic ...
                if block_type == 'ffn':
                    if layer_key == 'activation':
                        layer_key = 'relu'
                    organized[block_type][int(d['index'])][layer_key] = params
                    continue
                
                if layer_key == 'out_proj':
                    organized[block_type][int(d['index'])]['av_matmul'] = {
                        'type': 'MatMul_AV',
                        'input': None,
                        'output': params['input'][0]
                    }
                
                if layer_key == 'custom_softmax':
                    layer_key = 'softmax'
                    organized[block_type][int(d['index'])]['qk_matmul'] = {
                        'type': 'MatMul_QK', 
                        'input': None, 
                        'output': params['input'][0]
                    }
                
                organized[block_type][int(d['index'])][layer_key] = params
    
    return organized

def calculate_hw_params(effective_scale):
    if effective_scale == 0: return 0, 0
    mantissa, exponent = np.frexp(effective_scale)
    multiplier = int(round(mantissa * 2**8))
    shift = 8 - exponent
    return multiplier, shift


def dequantize_bias_to_int32(layer_params):
    """
    Converts a float bias to its effective int32 value. This is a critical step.
    The formula is: B_int32 = round(B_float / (S_input * S_weight))
    """
    bias_float = layer_params['bias']
    if bias_float is None:
        return np.zeros(layer_params['weight']['q_value'].shape[0], dtype=np.int32)
    s_act = layer_params['input'][0]['scale']
    s_weight = layer_params['weight']['scale']
    s_accumulator = s_act * s_weight
    effective_bias = np.round(bias_float / s_accumulator)
    return effective_bias.astype(np.int32)

def translate_torch_scales_to_hw_params(attn_block, ffn_block, H):
    hw_params = {}
    all_layers = {**attn_block, **ffn_block}
    
    for key, layer in all_layers.items():
        if layer.get('type') == 'Linear':
            # <-- FIX 5: Access the first input tensor for scale calculation
            effective_scale = (layer['input'][0]['scale'] * layer['weight']['scale']) / layer['output_scale']
            mult, shift = calculate_hw_params(effective_scale)
            current_H = 1 if key in ['fc1', 'fc2'] else H
            hw_params[key] = {'mult': [mult] * current_H, 'shift': [shift] * current_H, 'add': [layer['output_zp']] * current_H}

        elif layer.get('type') in ['MatMul_QK', 'MatMul_AV']:
            # The output scale/zp are already stored in the 'output' dict
            output_scale = layer['output']['scale']
            output_zp = layer['output']['zp']

            # <-- FIX 6: Use the correct input scales now that we capture both
            if key == 'qk_matmul':
                # Input to qk_matmul is output of q_proj and k_proj
                input1_scale = attn_block['q_proj']['output']['scale']
                input2_scale = attn_block['k_proj']['output']['scale']
            elif key == 'av_matmul':
                # Input to av_matmul is output of softmax and v_proj
                input1_scale = attn_block['softmax']['output']['scale']
                input2_scale = attn_block['v_proj']['output']['scale']
            
            effective_scale = (input1_scale * input2_scale) / output_scale
            mult, shift = calculate_hw_params(effective_scale)
            hw_params[key] = {'mult': [mult] * H, 'shift': [shift] * H, 'add': [output_zp] * H}

    if 'relu' in ffn_block:
        # Check if the neighbors (fc1 and fc2) were captured
        if 'fc1' in ffn_block and 'fc2' in ffn_block:
            # Input to ReLU is the output of fc1
            # NOTE: fc1's output is captured by its OWN hook.
            s_relu_in = ffn_block['fc1']['output']['scale']
            
            # Output of ReLU is the input to fc2
            # NOTE: fc2's input is captured by ITS hook.
            s_relu_out = ffn_block['fc2']['input'][0]['scale']

            effective_scale_relu = s_relu_in / s_relu_out if s_relu_out > 0 else 0
            mult_relu, shift_relu = calculate_hw_params(effective_scale_relu)
            
            # The 'add' for ReLU is the zero point of its OUTPUT tensor
            # We get this from the hook attached to the ReLU module itself.
            add_relu = ffn_block['relu']['output']['zp']

            hw_params['relu'] = {
                'mult': mult_relu,
                'shift': shift_relu,
                'add': add_relu,
                'q_1': 0, 'q_b': 0, 'q_c': 0 
            }
        else:
            print("⚠️ WARNING: Cannot derive ReLU params because fc1 or fc2 data is missing.")

    return hw_params


def verify_step(step_name, ita_result, torch_truth_dict):
    if torch_truth_dict is None or 'q_value' not in torch_truth_dict:
        print(f"\n🔬 Verifying: {step_name}\n  ⚠️ SKIPPED. PyTorch ground truth not found.")
        return False
    
    torch_truth = torch_truth_dict['q_value']
    
    # Squeeze extra dimensions from the PyTorch tensor until its rank matches the ITA result.
    # This handles the (B, H, S, S) vs (H, S, S) mismatch.
    while torch_truth.ndim > ita_result.ndim:
        torch_truth = np.squeeze(torch_truth, axis=0)

    if ita_result.shape != torch_truth.shape:
        print(f"\n🔬 Verifying: {step_name}\n  ❌ FAILED. Shape mismatch! ITA: {ita_result.shape}, PyTorch: {torch_truth.shape}")
        return False

    mae = np.mean(np.abs(ita_result.astype(np.int32) - torch_truth.astype(np.int32)))
    
    print(f"\n🔬 Verifying: {step_name}")
    if mae <= 1.0:
        print(f"  ✅ PASSED. MAE = {mae:.4f}")
        return True
    else:
        print(f"  ❌ FAILED. MAE = {mae:.4f}")
        print(f"    ITA:     min={np.min(ita_result)}, max={np.max(ita_result)}, mean={np.mean(ita_result):.2f}")
        print(f"    PyTorch: min={np.min(torch_truth)}, max={np.max(torch_truth)}, mean={np.mean(torch_truth):.2f}")
        return False

# ==============================================================================
# 3. MAIN VERIFICATION LOGIC (WITH CORRECTIONS)
# ==============================================================================

def main(args):
    
    output_base_location = os.path.join(PROJECT_ROOT, "third_party", "ITA_FPGA", "simvectors")

    # 2. Define the parameters that will be used for the folder name
    dims = {'S': 128, 'P': 192, 'E': 64, 'F': 256, 'H': 1}
    activation_name = "relu"
    bias_enabled = True # Transformer class default is bias=True

    # 3. Create the dynamic main folder name from the parameters
    main_folder_name = (f"data_S{dims['S']}_E{dims['E']}_P{dims['P']}_"
                        f"F{dims['F']}_H{dims['H']}_B{int(bias_enabled)}_{activation_name.capitalize()}")

    # 4. Create the full path for this specific simulation run
    main_output_path = os.path.join(output_base_location, main_folder_name)
    
    # --- Step 1: Create a "scaffold" of the final converted model ---
    print("--- Loading and Preparing PyTorch QAT Model ---")
    print(f"Instantiating model with {args.num_layers} layer(s).")
    model_qat = ITALSTMNetVIT_QAT(num_layers=args.num_layers)
    torch.backends.quantized.engine = 'qnnpack'
    
    # For perfect reproducibility, ensure the qconfig is set, just like in your training/inspection scripts
    # (Although it's likely already set in your model definition, this is safer)
    from torch.ao.quantization import QConfig, FusedMovingAvgObsFakeQuantize
    ita_symmetric_qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False
        ),
        weight=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False
        )
    )
    model_qat.attention_blocks.qconfig = ita_symmetric_qconfig
    model_qat.ffn_blocks.qconfig = ita_symmetric_qconfig

    # Prepare and immediately convert the model to get the final architecture
    model_prepared = torch.ao.quantization.prepare_qat(model_qat.train())
    model_converted = torch.ao.quantization.convert(model_prepared.eval())

    # --- Step 2: Load the state_dict from the saved CONVERTED model ---
    # The traceback proves the .pth file is from a converted model.
    print("Loading saved state_dict into the converted model...")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    
    # THE KEY FIX: Use strict=False to ignore minor key mismatches,
    # which is common when loading converted models. This will load all
    # the important _packed_params and quantization parameters that do match.
    model_converted.load_state_dict(state_dict, strict=False)
    
    model_converted.eval() # Ensure model is in eval mode for analysis
    print("✅ PyTorch Model is ready for analysis.")
    
    print("\n" + "="*80)
    print("🔬 Inspecting modules in the final converted model...")
    print("="*80)
    for name, module in model_converted.named_modules():
        # We are interested in the FFN block specifically
        if 'ffn_blocks.0' in name:
            print(f"Name: {name:<40} | Type: {type(module)}")
    print("="*80 + "\n")
    
    
        
    print("Patching matmul2 to handle mixed dtype inputs...")
    for block in model_converted.attention_blocks:
        # After conversion, matmul2 is a QFunctional module with a learned scale/zp
        matmul2_module = block.matmul2
        learned_output_scale = matmul2_module.scale
        learned_output_zero_point = matmul2_module.zero_point
        
        # Create a lambda to wrap our custom function with the learned params
        new_matmul_func = lambda q_a, q_b: custom_quantized_matmul(
            q_a, q_b,
            output_scale=learned_output_scale,
            output_zero_point=learned_output_zero_point
        )
        # Overwrite the matmul method of the QFunctional module
        matmul2_module.matmul = new_matmul_func
    print("✅ matmul2 patched.")

    # --- Step 3: Prepare input for the forward pass ---
    img = Image.open(args.image).convert('L')
    preprocess = transforms.Compose([transforms.Resize((60, 90)), transforms.ToTensor()])
    img_tensor = preprocess(img).unsqueeze(0)
    dummy_input = [img_tensor, torch.randn(1, 1), torch.randn(1, 4), None]

    # --- Step 4: Extract all parameters and run inference (The rest is the same) ---
    extractor = PyTorchDataExtractor(model_converted)
    all_torch_params = extractor.extract_and_capture(dummy_input)
    
    print("\n--- DEBUG: Keys captured by the extractor ---")
    for key in sorted(all_torch_params.keys()):
        print(key)
    print("--- END DEBUG ---\n")

    print("\n--- Step 5: Organizing Parameters by Block ---")
    organized_params = organize_params_by_block(all_torch_params)
    print("✅ Parameters organized.")
    
    # --- Loop over blocks for verification (No changes needed below this line) ---
    for block_index in sorted(organized_params['attention'].keys()):
        print("\n" + "="*80)
        print(f"     VERIFYING TRANSFORMER BLOCK {block_index} vs. PYTORCH GROUND TRUTH")
        print("="*80)

        block_folder_name = f"pytorch_test_vectors_block_{block_index}"
        output_dir = os.path.join(main_output_path, block_folder_name)
        
        print(f"✅ Writing output for block {block_index} to: {os.path.abspath(output_dir)}")
        
        attn_block = organized_params['attention'][block_index]
        ffn_block = organized_params['ffn'][block_index]
        
        
        
        #import pprint
        #print("--- DEBUG: Organized Attention Block Contents ---")
        #pprint.pprint(attn_block)
        #print("--- DEBUG: Organized FFN Block Contents ---")
        #pprint.pprint(ffn_block)

        def get_and_squeeze(param_dict):
            if param_dict is None or 'q_value' not in param_dict: return None
            val = param_dict['q_value']
            
            # <-- CHANGE: Squeeze the batch dimension (axis 0)
            if val.ndim >= 1 and val.shape[0] == 1:
                val = np.squeeze(val, axis=0)

            # <-- CHANGE: If it's still a 3D tensor (e.g., (1, 128, 128) for H=1), squeeze the head dimension (axis 0 again)
            # This handles the specific case of attention layers where we expect (Seq, Seq) or (Seq, Embed)
            if val.ndim == 3 and val.shape[0] == 1:
                val = np.squeeze(val, axis=0)

            return val

        block_tensors = {
            'Q': get_and_squeeze(attn_block['q_proj']['input'][0]),
            'K': get_and_squeeze(attn_block['k_proj']['input'][0]),
            'V': get_and_squeeze(attn_block['v_proj']['input'][0]),
            'FF_in': get_and_squeeze(ffn_block['fc1']['input'][0]),

            # <-- CHANGE: Add .T to transpose each weight matrix
            'Wq': np.expand_dims(attn_block['q_proj']['weight']['q_value'].T, axis=0),
            'Wk': np.expand_dims(attn_block['k_proj']['weight']['q_value'].T, axis=0),
            'Wv': np.expand_dims(attn_block['v_proj']['weight']['q_value'].T, axis=0),
            'Wo': np.expand_dims(attn_block['out_proj']['weight']['q_value'].T, axis=0),
            'Wff': np.expand_dims(ffn_block['fc1']['weight']['q_value'].T, axis=0),
            'Wff2': np.expand_dims(ffn_block['fc2']['weight']['q_value'].T, axis=0),

            # The biases are 1D vectors, so they do not need to be transposed.
            'Bq': np.expand_dims(dequantize_bias_to_int32(attn_block['q_proj']), axis=0),
            'Bk': np.expand_dims(dequantize_bias_to_int32(attn_block['k_proj']), axis=0),
            'Bv': np.expand_dims(dequantize_bias_to_int32(attn_block['v_proj']), axis=0),
            'Bo': np.expand_dims(dequantize_bias_to_int32(attn_block['out_proj']), axis=0),
            'Bff': np.expand_dims(dequantize_bias_to_int32(ffn_block['fc1']), axis=0),
            'Bff2': np.expand_dims(dequantize_bias_to_int32(ffn_block['fc2']), axis=0),
        }

        
        hw_params = translate_torch_scales_to_hw_params(attn_block, ffn_block, H=dims['H'])

        ita_sim = Transformer(ITA_N=16, path=output_dir, activation='relu', **dims, **block_tensors, quant_params=hw_params)
        
        ita_sim.step1_Qp()
        verify_step("Q Projection", ita_sim.Qp_requant, attn_block['q_proj']['output'])
        ita_sim.step2_Kp()
        verify_step("K Projection", ita_sim.Kp_requant, attn_block['k_proj']['output'])
        ita_sim.step3_Vp()
        verify_step("V Projection", ita_sim.Vp_requant, attn_block['v_proj']['output'])
        
        ita_sim.step4_QK(no_partial_softmax=False)
        verify_step("QK MatMul (Logits)", ita_sim.A_requant, attn_block['qk_matmul']['output'])
        verify_step("Softmax (Integer Approx)", ita_sim.A_partial_softmax, attn_block['softmax']['output'])

        if args.isolate_softmax:
            print("  ℹ️  --isolate_softmax flag is active. Injecting PyTorch ground truth to test subsequent layers.")
            softmax_truth = attn_block['softmax']['output']['q_value']
            # Squeeze truth to match ITA's (H, S, S) shape
            while softmax_truth.ndim > ita_sim.A_partial_softmax.ndim:
                softmax_truth = np.squeeze(softmax_truth, axis=0)
            ita_sim.A_partial_softmax = softmax_truth
        
        ita_sim.step5_AV()
        verify_step("AV MatMul (Context)", ita_sim.O_soft_requant, attn_block['av_matmul']['output'])
        ita_sim.step6_O()
        verify_step("Output Projection", ita_sim.Out_soft_requant, attn_block['out_proj']['output'])

        ita_sim.feedforward_layer()
        # We cannot verify the intermediate result after the first requantization
        # because PyTorch fuses the Linear+ReLU and doesn't expose that tensor.
        print("\n🔬 Verifying: FFN Layer 1 (pre-activation)")
        #print(f"  ⚠️ SKIPPED. This intermediate tensor is not available in fused PyTorch model.")

        # We verify the final result of the fused operation. In the simulator,
        # FFp_requant is updated in-place by the apply_activation call.
        verify_step("FFN Layer 1 + ReLU", ita_sim.FFp_requant, ffn_block['relu']['output'])
        verify_step("FFN Layer 2", ita_sim.FF2p_requant, ffn_block['fc2']['output'])
        
        ita_sim.export_hwpe()

    print("\n" + "="*80)
    print("Verification run complete.")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract PyTorch QAT data and run through the ITA hardware model for verification.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the final quantized PyTorch model state_dict (.pth)')
    parser.add_argument('--image', type=str, required=True, help='Path to the ground-truth input image.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of transformer layers in the model to be verified.')
    parser.add_argument('--isolate_softmax', action='store_true', help='Inject the PyTorch ground truth after softmax to isolate its error.')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Error: Checkpoint file not found at {args.checkpoint}")
    if not os.path.exists(args.image):
        sys.exit(f"Error: Image file not found at {args.image}")

    main(args)