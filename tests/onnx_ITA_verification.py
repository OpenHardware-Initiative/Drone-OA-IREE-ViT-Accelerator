# final_verification.py (Refactored for Robustness)

import numpy as np
import argparse
import os
import sys
import re
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.ITA_FPGA.PyITA.ITA import Transformer
from tests.onnx_param_extractor import ONNXParameterExtractor, get_ground_truth_input, run_and_capture_intermediates

## --- KEY CHANGE: New helper function to organize the flat list of parameters ---
def organize_params_by_block(static_params):
    """
    Re-organizes the flat list of parameters from the extractor into a structured
    dictionary, organized by block type and index.
    """
    organized = defaultdict(lambda: defaultdict(dict))
    
    # Regex for Linear layers (attention and ffn)
    linear_pattern = re.compile(r"/(?P<type>attention_blocks|ffn_blocks)\.(?P<index>\d+)/(?P<layer>[\w_]+)")
    # Regex for activation-only MatMuls
    matmul_pattern = re.compile(r"/(?P<type>attention_blocks)\.(?P<index>\d+)/(?P<name>MatMul(_1)?)")
    # Regex for Softmax/ReLU layers
    activation_pattern = re.compile(r"/(?P<type>attention_blocks|ffn_blocks)\.(?P<index>\d+)/.+/(?P<op>Softmax|Relu|ReLU)")

    for layer_type, params_list in static_params.items():
        if layer_type == 'Linear':
            for p in params_list:
                match = linear_pattern.search(p['name'])
                if match:
                    d = match.groupdict()
                    block_type = 'attention' if 'attention' in d['type'] else 'ffn'
                    organized[block_type][int(d['index'])][d['layer']] = p
        
        elif layer_type == 'MatMul_Activation':
            for p in params_list:
                match = matmul_pattern.search(p['name'])
                if match:
                    d = match.groupdict()
                    key = 'qk_matmul' if d['name'] == 'MatMul' else 'av_matmul'
                    if 'MatMul_Activation' not in organized['attention'][int(d['index'])]:
                        organized['attention'][int(d['index'])]['MatMul_Activation'] = {}
                    organized['attention'][int(d['index'])]['MatMul_Activation'][key] = p

        elif layer_type == 'Softmax' or layer_type == 'ReLU':
             for p in params_list:
                match = activation_pattern.search(p['name'])
                if match:
                    d = match.groupdict()
                    block_type = 'attention' if 'attention' in d['type'] else 'ffn'
                    op_name = d['op'].lower().replace('relu', 'relu') # Normalize name
                    organized[block_type][int(d['index'])][op_name] = p

    return organized


def calculate_multiplier_shift(effective_scale, bit_width=31):
    """Calculates the integer multiplier and bit-shift required for requantization."""
    if isinstance(effective_scale, np.ndarray):
        multiplier = np.round(effective_scale * (2**bit_width)).astype(np.int64)
    else:
        multiplier = int(round(effective_scale * (2**bit_width)))
    return multiplier, bit_width

def calculate_hw_params(effective_scale):
    """
    Calculates the 8-bit multiplier and shift that PyITA expects.
    This emulates the library's internal logic.
    """
    if effective_scale == 0:
        return 0, 0
    
    # Decompose the scale into a normalized mantissa and an exponent
    # effective_scale = mantissa * 2**exponent, where 0.5 <= mantissa < 1.0
    mantissa, exponent = np.frexp(effective_scale)
    
    # The PyITA library expects an 8-bit multiplier
    multiplier = int(round(mantissa * 2**8))
    
    # The corresponding shift is derived from the exponent
    shift = 8 - exponent
    
    return multiplier, shift

## --- KEY CHANGE: This function is now simpler and more robust ---
def translate_onnx_scales_to_hw_params(block_params, ffn_global_params, H):
    """Converts ONNX scales into the PyITA hardware's 8-bit mult/shift format."""
    hw_params = {}
    key_map = {
        'q_proj': 'q_proj', 'k_proj': 'k_proj', 'v_proj': 'v_proj',
        'out_proj': 'out_proj', 'fc1': 'fc1', 'fc2': 'fc2'
    }

    # Process Linear layers
    for onnx_key, hw_key in key_map.items():
        if onnx_key in block_params:
            layer = block_params[onnx_key]
            current_H = 1 if onnx_key in ['fc1', 'fc2'] else H
            effective_scale = (layer['activation']['scale_value'] * layer['weight']['scale_value']) / layer['output_scale']
            mult, shift = calculate_hw_params(effective_scale)
            hw_params[hw_key] = {'mult': [mult] * current_H, 'shift': [shift] * current_H, 'add': [layer['output_zp']] * current_H}

    # Process Activation-only MatMuls
    if 'MatMul_Activation' in block_params:
        for matmul_key, matmul_layer in block_params['MatMul_Activation'].items():
            effective_scale = (matmul_layer['activation']['scale_value'] * matmul_layer['weights']['scale_value']) / matmul_layer['output_scale']
            mult, shift = calculate_hw_params(effective_scale)
            hw_params[matmul_key] = {'mult': [mult] * H, 'shift': [shift] * H, 'add': [matmul_layer['output_zp']] * H}
            
    # Add ReLU parameters
    if ffn_global_params.get('relu'):
        relu_layer = ffn_global_params['relu']
        effective_scale = relu_layer['activation']['scale_value'] / relu_layer['output_scale']
        mult, shift = calculate_hw_params(effective_scale)
        hw_params['relu'] = {'mult': mult, 'shift': shift, 'add': relu_layer['output_zp'], 'q_1': 0, 'q_b': 0, 'q_c': 0}
            
    return hw_params


def verify_step(step_name, ita_result, onnx_truth):
    """Helper function to compare two tensors and print the result."""
    if onnx_truth is None:
        print(f"\n🔬 Verifying: {step_name}")
        print(f"  ⚠️ SKIPPED. ONNX ground truth not found.")
        return False
        
    # Squeeze out batch/head dimensions for a fair comparison
    ita_result_flat = ita_result.flatten()
    onnx_truth_flat = onnx_truth.flatten()

    mae = np.mean(np.abs(ita_result_flat.astype(np.int32) - onnx_truth_flat.astype(np.int32)))
    
    print(f"\n🔬 Verifying: {step_name}")
    if mae <= 1.0: # Allow an average error of 1 due to potential rounding differences
        print(f"  ✅ PASSED. MAE = {mae:.4f}")
        return True
    else:
        print(f"  ❌ FAILED. MAE = {mae:.4f}")
        # To debug, you can print the tensors:
        print(f"ITA: max result {max(ita_result_flat)}, min result {min(ita_result_flat)}")
        print(f"ONNX: max result {max(onnx_truth_flat)}, min result {min(onnx_truth_flat)}")
        return False

def manual_quantize(float_tensor, scale, zero_point, dtype=np.uint8):
    """
    Manually quantizes a float tensor to an integer tensor.
    """
    # The formula is: q = round(real / scale) + zp
    quantized_float = np.round(float_tensor / scale) + zero_point
    
    # Clip the values to the valid range of the target integer type
    if dtype == np.uint8:
        q_min, q_max = 0, 255
    else: # np.int8
        q_min, q_max = -128, 127
        
    clipped_tensor = np.clip(quantized_float, q_min, q_max)
    
    return clipped_tensor.astype(dtype)

def main(args):
    # 1. Extract all static parameters and runtime values from ONNX model
    print("--- Step 1: Extracting Parameters from ONNX Model ---")
    extractor = ONNXParameterExtractor(args.onnx_model)
    all_static_params = extractor.extract_all_parameters()
    
    # Re-organize the flat data into a structured, per-block dictionary
    organized_params = organize_params_by_block(all_static_params)
    print("✅ Parameters organized by block.")

    # Run inference once to capture all intermediate activation tensors
    print("\n--- Step 2: Running Inference to Capture Ground Truth Activations ---")
    input_image_np = get_ground_truth_input(args.image)
    input_feed = {"img_data": input_image_np, "additional_data": np.zeros((1, 1), dtype=np.float32), "quat_data": np.zeros((1, 4), dtype=np.float32), "h_in": np.zeros((3, 1, 128), dtype=np.float32), "c_in": np.zeros((3, 1, 128), dtype=np.float32)}
    names_to_capture = {p['output_q_tensor'] for op_list in all_static_params.values() for p in op_list}
    names_to_capture.update({p['activation']['q_tensor_name'] for op_list in all_static_params.values() for p in op_list if 'activation' in p})
    for p in all_static_params.get('Softmax', []):
        if 'float_output_tensor' in p:
            names_to_capture.add(p['float_output_tensor'])
    # CORRECTED Line in onnx_ITA_verification.py
    all_runtime_values = run_and_capture_intermediates(args.onnx_model, input_feed, list(names_to_capture))

    ## --- KEY CHANGE: Loop over all found blocks instead of hardcoding one ---
    for block_index in sorted(organized_params['attention'].keys()):
        print("\n" + "="*80)
        print(f"     VERIFYING TRANSFORMER BLOCK {block_index} vs. ONNX GROUND TRUTH")
        print("="*80)

        attn_block = organized_params['attention'][block_index]
        ffn_block = organized_params['ffn'][block_index]
        
        def get_and_squeeze(tensor_name):
            tensor = all_runtime_values.get(tensor_name)
            return np.squeeze(tensor, axis=0) if tensor is not None else None
        
        def dequantize_bias_to_int32(layer_params):
            """Converts a stored quantized bias to its effective int32 value."""
            # Use float64 for precision during the calculation
            q_bias = layer_params['bias']['q_value'].astype(np.float64)
            s_bias = layer_params['bias']['scale_value']
            s_act = layer_params['activation']['scale_value']
            s_weight = layer_params['weight']['scale_value']

            # This is the scale of the int32 matmul accumulator
            s_accumulator = s_act * s_weight
            
            # Dequantize bias to float and requantize to the accumulator's scale
            effective_bias = np.round((q_bias * s_bias) / s_accumulator)
            
            return effective_bias.astype(np.int32)

        block_tensors = {
            'Q': get_and_squeeze(attn_block['q_proj']['activation']['q_tensor_name']),
            'K': get_and_squeeze(attn_block['k_proj']['activation']['q_tensor_name']),
            'V': get_and_squeeze(attn_block['v_proj']['activation']['q_tensor_name']),
            'FF_in': get_and_squeeze(ffn_block['fc1']['activation']['q_tensor_name']),
            
            'Wq': np.expand_dims(attn_block['q_proj']['weight']['q_value'], axis=0),
            'Wk': np.expand_dims(attn_block['k_proj']['weight']['q_value'], axis=0),
            'Wv': np.expand_dims(attn_block['v_proj']['weight']['q_value'], axis=0),
            'Wo': np.expand_dims(attn_block['out_proj']['weight']['q_value'], axis=0),
            'Wff': np.expand_dims(ffn_block['fc1']['weight']['q_value'], axis=0),
            'Wff2': np.expand_dims(ffn_block['fc2']['weight']['q_value'], axis=0),

            # Dequantize biases to their effective int32 values before adding the head dimension
            'Bq': np.expand_dims(dequantize_bias_to_int32(attn_block['q_proj']), axis=0),
            'Bk': np.expand_dims(dequantize_bias_to_int32(attn_block['k_proj']), axis=0),
            'Bv': np.expand_dims(dequantize_bias_to_int32(attn_block['v_proj']), axis=0),
            'Bo': np.expand_dims(dequantize_bias_to_int32(attn_block['out_proj']), axis=0),
            'Bff': np.expand_dims(dequantize_bias_to_int32(ffn_block['fc1']), axis=0),
            'Bff2': np.expand_dims(dequantize_bias_to_int32(ffn_block['fc2']), axis=0),
        }

        dims = {'S': 128, 'P': 192, 'E': 128, 'F': 256, 'H': 1}
        
        # The translate function now works on a single, organized block's data
        hw_params = translate_onnx_scales_to_hw_params({**attn_block, **ffn_block}, ffn_global_params=organized_params['ffn'], H=dims['H'])

        # Add ReLU params if they exist
        if 'relu' in organized_params['ffn']:
            relu_layer = organized_params['ffn']['relu']
            effective_scale = relu_layer['activation']['scale_value'] / relu_layer['output_scale']
            mult, shift = calculate_multiplier_shift(effective_scale)
            hw_params['relu'] = {'mult': mult, 'shift': shift, 'add': relu_layer['output_zp'], 'q_1': 0, 'q_b': 0, 'q_c': 0}

        # 4. Instantiate and run your hardware simulation class
        
        output_dir = f"onnx_test_vectors_block_{block_index}"
        ita_sim = Transformer(ITA_N=16, path=output_dir, activation='relu', **dims, **block_tensors, quant_params=hw_params)

        # 5. Run simulation & verification step-by-step
        ita_sim.step1_Qp()
        verify_step("Q Projection", ita_sim.Qp_requant, all_runtime_values.get(attn_block['q_proj']['output_q_tensor']))

        ita_sim.step2_Kp()
        verify_step("K Projection", ita_sim.Kp_requant, all_runtime_values.get(attn_block['k_proj']['output_q_tensor']))

        ita_sim.step3_Vp()
        verify_step("V Projection", ita_sim.Vp_requant, all_runtime_values.get(attn_block['v_proj']['output_q_tensor']))
        
        ita_sim.step4_QK(no_partial_softmax=False)
        print(f"Data type of QK MatMul output: {ita_sim.A_requant.dtype}, ONNX type: {all_runtime_values.get(attn_block['MatMul_Activation']['qk_matmul']['output_q_tensor']).dtype}")
        verify_step("QK MatMul (A_requant)", ita_sim.A_requant, all_runtime_values.get(attn_block['MatMul_Activation']['qk_matmul']['output_q_tensor']))
        
        forced_scale = 1.0 / 128.0
        forced_zero_point = 127

        # Get the float32 ground truth from the ONNX model run
        onnx_softmax_float_output = all_runtime_values.get(attn_block['softmax']['float_output_tensor'])
        onnx_softmax_forced_uint8 = manual_quantize(onnx_softmax_float_output, forced_scale, forced_zero_point)
        verify_step("Softmax (A_partial_softmax vs. Forced Quant)", ita_sim.A_partial_softmax, onnx_softmax_forced_uint8)
        
        print(f"Data type of A_requant: {ita_sim.A_partial_softmax.dtype}, ONNX type: {all_runtime_values.get(attn_block['MatMul_Activation']['qk_matmul']['output_q_tensor']).dtype}")
        verify_step("Softmax (A_partial_softmax)", ita_sim.A_partial_softmax, all_runtime_values.get(attn_block['softmax']['output_q_tensor']))

        if args.isolate_softmax:
            print("  ℹ️  --isolate_softmax flag is active. Injecting ONNX ground truth to test subsequent layers.")
            ita_sim.A_partial_softmax = get_and_squeeze(attn_block['softmax']['output_q_tensor'])
        
        ita_sim.step5_AV()
        verify_step("AV MatMul (O_soft)", ita_sim.O_soft_requant, all_runtime_values.get(attn_block['MatMul_Activation']['av_matmul']['output_q_tensor']))

        ita_sim.step6_O()
        verify_step("Output Projection", ita_sim.Out_soft_requant, all_runtime_values.get(attn_block['out_proj']['output_q_tensor']))

        # --- FEED FORWARD NETWORK VERIFICATION ---
        ita_sim.feedforward_layer()
        verify_step("FFN Layer 1 + ReLU (FFp_requant)", ita_sim.FFp_requant, all_runtime_values.get(ffn_block['relu']['output_q_tensor']))
        verify_step("FFN Layer 2 (FF2p_requant)", ita_sim.FF2p_requant, all_runtime_values.get(ffn_block['fc2']['output_q_tensor']))

        
    print("\n" + "="*80)
    print("Verification run complete.")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract ONNX data and run through the ITA hardware model.")
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to the final quantized ONNX model (.onnx)')
    parser.add_argument('--image', type=str, required=True, help='Path to the ground-truth input image.')
    parser.add_argument('--isolate_softmax', action='store_true', help='Inject the ONNX ground truth after softmax to isolate its error.')
    args = parser.parse_args()
    main(args)