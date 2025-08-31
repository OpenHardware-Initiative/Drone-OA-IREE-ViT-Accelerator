import torch
import torch.quantization
import argparse
import sys
import os

# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Use the correct import path for your model
from models.ITA_single_layer_upsample_shuffle.QAT.model import ITALSTMNetVIT_QAT

def check_model_quantization(checkpoint_path):
    """
    Loads a converted QAT model and inspects its weights and input quantizer.
    """
    print(f"---  inspecting model: {checkpoint_path} ---")

    # 1. Instantiate the FLOATING-POINT model architecture
    model = ITALSTMNetVIT_QAT(params={}, qat_mode=False)
    model.eval()

    # --- FIX: Use the correct preparation and conversion workflow ---
    # This correctly creates the final integer model structure in memory.
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model_prepared = torch.quantization.prepare(model)
    model_converted = torch.quantization.convert(model_prepared)
    
    # 2. Now, load the state_dict into the correctly structured model
    # To address the warning, we use weights_only=True.
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_converted.load_state_dict(state_dict)
    
    # Use the converted model for the rest of the script
    model = model_converted
    
    # --- Part 1: Check the Weights ---
    print("\n## 1. Checking Layer Weights...")
    q_proj_layer = model.attention_blocks[0].q_proj
    
    # For quantized layers, we use weight() to get the quantized tensor
    quantized_weight = q_proj_layer.weight()
    
    # .int_repr() gives the underlying 8-bit integer tensor
    weight_int_repr = quantized_weight.int_repr()
    
    print(f"Weight Data Type: {weight_int_repr.dtype}")
    print(f"Weight Value Range: Min={weight_int_repr.min().item()}, Max={weight_int_repr.max().item()}")

    # --- Part 2: Check the Input Preparation ---
    print("\n## 2. Checking Model Input Quantizer...")
    input_quantizer = model.quant_attention[0]
    print(f"Input Quantizer Scale: {input_quantizer.scale}")
    print(f"Input Quantizer Zero Point: {input_quantizer.zero_point}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model_quantized_final.pth file')
    args = parser.parse_args()
    check_model_quantization(args.checkpoint)