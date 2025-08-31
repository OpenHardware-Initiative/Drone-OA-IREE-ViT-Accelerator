import torch
import torch.ao.quantization
import argparse
import sys
import os
import pprint
from PIL import Image
from torchvision import transforms
from torch.ao.quantization import QConfig, FusedMovingAvgObsFakeQuantize


# --- Ensure correct paths for imports ---
# Adjust this path if your script is in a different location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import your QAT model definition
from models.ITA_single_layer_upsample_shuffle.QAT.model import ITALSTMNetVIT_QAT

# ==============================================================================
# Helper Functions for Inspection
# ==============================================================================

# This dictionary will store all the captured tensor values
captured_values = {}

def get_hook(name):
    """Factory function to create a forward hook for a specific module."""
    def hook(model, input, output):
        # Store both input and output tuples
        captured_values[name] = {
            'input': input,
            'output': output
        }
    return hook

def print_tensor_summary(label, tensor):
    """Prints a formatted summary of a tensor."""
    if tensor is None:
        print(f"  - {label:<25} | Not found or None")
        return
        
    # Dequantize if it's a quantized tensor to see float range
    if tensor.is_quantized:
        tensor_float = tensor.dequantize()
        min_val_f = f"{tensor_float.min().item():<8.4f}"
        max_val_f = f"{tensor_float.max().item():<8.4f}"
        int_repr = tensor.int_repr()
        min_val_i = f"{int_repr.min().item():<5}"
        max_val_i = f"{int_repr.max().item():<5}"
        q_params = f"Scale={tensor.q_scale():.4e}, ZP={tensor.q_zero_point()}"
    else:
        min_val_f, max_val_f = f"{tensor.min().item():<8.4f}", f"{tensor.max().item():<8.4f}"
        min_val_i, max_val_i = "N/A", "N/A" # Not an integer tensor
        q_params = "N/A (Float Tensor)"

    print(f"  - {label:<25} | Shape={str(tensor.shape):<20} | DType={str(tensor.dtype):<15} | "
          f"Int Range=[{min_val_i}, {max_val_i}] | Float Range=[{min_val_f}, {max_val_f}]")
    print(f"    {'':<27} | {q_params}")


def custom_quantized_matmul(q_a: torch.Tensor, 
                              q_b: torch.Tensor, 
                              output_scale: float, 
                              output_zero_point: int) -> torch.Tensor:
    # ... (the exact code you provided) ...
    assert q_a.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    assert q_b.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    a_int = q_a.int_repr().to(torch.int32)
    b_int = q_b.int_repr().to(torch.int32)
    a_scale, b_scale = q_a.q_scale(), q_b.q_scale()
    a_zero_point, b_zero_point = q_a.q_zero_point(), q_b.q_zero_point()
    accumulator = torch.matmul(a_int - a_zero_point, b_int - b_zero_point)
    requantization_multiplier = (a_scale * b_scale) / output_scale
    output_unclamped = accumulator.float() * requantization_multiplier + output_zero_point
    q_out_clamped = output_unclamped.round().clamp(-128, 127).to(torch.int8)
    return torch._make_per_tensor_quantized_tensor(
        q_out_clamped, scale=output_scale, zero_point=output_zero_point
    )

# ==============================================================================
# Main Inspection Logic
# ==============================================================================

def inspect_model(checkpoint_path, image_path):
    print(f"--- Inspecting Quantized Model: {checkpoint_path} ---")

    # 1. Prepare the model structure to match the saved state_dict
    print("Instantiating and converting model structure...")
    model_qat = ITALSTMNetVIT_QAT()
    
    torch.backends.quantized.engine = 'qnnpack'
    
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

    model_prepared = torch.ao.quantization.prepare_qat(model_qat)
    model_converted = torch.ao.quantization.convert(model_prepared)
    model_converted.eval()  # Set to eval mode for inference

    # 2. Load the state_dict into the correctly structured model
    print("Loading saved state_dict...")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_converted.load_state_dict(state_dict)
    model = model_converted
    
    print("Registering hooks and patching matmul2...")
    for block_idx, block in enumerate(model.attention_blocks):
        # --- Patch matmul2 to use our custom function ---
        # After conversion, matmul2 is a QFunctional module with learned scale/zp
        matmul2_module = block.matmul2
        learned_output_scale = matmul2_module.scale
        learned_output_zero_point = matmul2_module.zero_point
        
        # We need to wrap our custom function in a lambda to pass the learned params
        new_matmul_func = lambda q_a, q_b: custom_quantized_matmul(
            q_a, 
            q_b, 
            output_scale=learned_output_scale,
            output_zero_point=learned_output_zero_point
        )
        # Overwrite the matmul method of the converted QFunctional module
        matmul2_module.matmul = new_matmul_func
        
        # --- Register hooks for all other modules of interest ---
        block.q_proj.register_forward_hook(get_hook(f'attn_{block_idx}_q_proj'))
        block.k_proj.register_forward_hook(get_hook(f'attn_{block_idx}_k_proj'))
        block.v_proj.register_forward_hook(get_hook(f'attn_{block_idx}_v_proj'))
        block.softmax.register_forward_hook(get_hook(f'attn_{block_idx}_softmax'))
        block.matmul1.register_forward_hook(get_hook(f'attn_{block_idx}_matmul_qk')) # Hook matmul1 to get logits
        block.out_proj.register_forward_hook(get_hook(f'attn_{block_idx}_out_proj'))
        
        ffn_block = model.ffn_blocks[block_idx]
        ffn_block.fc1.register_forward_hook(get_hook(f'ffn_{block_idx}_fc1'))
        ffn_block.activation.register_forward_hook(get_hook(f'ffn_{block_idx}_relu'))
        ffn_block.fc2.register_forward_hook(get_hook(f'ffn_{block_idx}_fc2'))
    print("✅ Hooks and patches are set.")


    # 4. Run forward pass
    print(f"Preparing input image from: {image_path}")
    img = Image.open(image_path).convert('L')
    preprocess = transforms.Compose([transforms.Resize((60, 90)), transforms.ToTensor()])
    img_tensor = preprocess(img).unsqueeze(0)
    dummy_input = [img_tensor, torch.randn(1, 1), torch.randn(1, 4), None]
    
    print("Running a single forward pass to capture values...")
    with torch.no_grad():
        model(dummy_input)
    print("✅ Values captured.")
    
    print("\n--- All Captured Keys ---")
    pprint.pprint(sorted(captured_values.keys()))
    print("-------------------------\n")

    # 5. Print the Detailed Report
    print("="*80)
    print("         COMPLETE QUANTIZATION INSPECTION REPORT")
    print("="*80)
    
    for i in range(len(model.attention_blocks)):
        print(f"\n{'#'*20} ATTENTION BLOCK {i} {'#'*20}")
        
        print("\n--- Q, K, V Projections ---")
        print_tensor_summary("Q Output", captured_values.get(f'attn_{i}_q_proj', {}).get('output'))
        print_tensor_summary("K Output", captured_values.get(f'attn_{i}_k_proj', {}).get('output'))
        print_tensor_summary("V Output", captured_values.get(f'attn_{i}_v_proj', {}).get('output'))
        
        print("\n--- Logits & Attention Weights (Custom Softmax) ---")
        # Get the dictionary for the custom_softmax hook
        softmax_capture = captured_values.get(f'attn_{i}_custom_softmax', {})

        # The logits are the INPUT to this module. The input is a tuple, so we get the first element.
        logits_quant = softmax_capture.get('input', (None,))[0]

        # The attention weights are the OUTPUT of this module.
        attn_weights_quant = softmax_capture.get('output')

        print_tensor_summary("Logits (Input)", logits_quant)
        print_tensor_summary("Attn Weights (Output)", attn_weights_quant)
        
        print_tensor_summary("Logits (from matmul1)", logits_quant)
        print_tensor_summary("Attn Weights (Output)", attn_weights_quant)

        print("\n--- Context Vector ---")
        context_quant = captured_values.get(f'attn_{i}_out_proj', {}).get('input', [None])[0]
        print_tensor_summary("Context (Quantized)", context_quant)

        print("\n--- Output Projection ---")
        out_proj_layer = model.attention_blocks[i].out_proj
        out_proj_output = captured_values.get(f'attn_{i}_out_proj', {}).get('output')
        print_tensor_summary("Input (Context)", context_quant)
        print_tensor_summary("Weight", out_proj_layer.weight())
        if out_proj_layer.bias() is not None:
            print_tensor_summary("Bias", out_proj_layer.bias())
        print_tensor_summary("Final Output", out_proj_output)
        
        if f'ffn_{i}_fc1' in captured_values:
            print("\n" + "#"*20 + f" FFN BLOCK {i} " + "#"*20)
            fc1_layer = model.ffn_blocks[i].fc1
            print("\n--- FFN FC1 Layer ---")
            print_tensor_summary("Input", captured_values[f'ffn_{i}_fc1']['input'][0])
            print_tensor_summary("Weight", fc1_layer.weight())
            if fc1_layer.bias() is not None: print_tensor_summary("Bias", fc1_layer.bias())
            print_tensor_summary("Output", captured_values[f'ffn_{i}_fc1']['output'])

            print("\n--- FFN ReLU ---")
            print_tensor_summary("Input", captured_values[f'ffn_{i}_relu']['input'][0])
            print_tensor_summary("Output", captured_values[f'ffn_{i}_relu']['output'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a converted QAT model's intermediate quantized values.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the final quantized model state_dict (.pth)')
    parser.add_argument('--image', type=str, required=True, help='Path to a sample input image for inference.')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
        
    inspect_model(args.checkpoint, args.image)
