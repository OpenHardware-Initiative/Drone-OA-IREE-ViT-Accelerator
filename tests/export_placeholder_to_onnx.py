# export_placeholder_to_onnx.py

import torch
import torch.nn as nn
from torch.quantization import QConfig, FusedMovingAvgObsFakeQuantize
import argparse 

import os
import sys
# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.ITA.export.ITA_model_export import ITALSTMNetVIT_Export
from models.ITA.dispatch.ITA_model_placeholder import ITALSTMNetVIT_Placeholder_Export
from utils.load_model import load_weights_to_placeholder_model, HardcodedQuantizer


def export_placeholder_model(trained_model_path: str):
    """Prepares and exports the ITALSTMNetVIT_Placeholder_Export model to ONNX."""
    
    print("--- Step 1: Load Trained Quantized Model ---")
    original_model = ITALSTMNetVIT_Export()
    
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
    
    original_model.qconfig = ita_symmetric_qconfig
    # Exclude layers from quantization
    original_model.tokenizer.qconfig = None
    for block in original_model.norm1_layers: block.qconfig = None
    for block in original_model.norm2_layers: block.qconfig = None
        
    original_model.train()
    torch.quantization.prepare_qat(original_model, inplace=True)
    loaded_quantized_model = torch.quantization.convert(original_model, inplace=False)
    
    if not os.path.exists(trained_model_path):
        print(f"❌ ERROR: Trained model checkpoint not found at '{trained_model_path}'.")
        return
    
    print(f"Loading trained quantized model weights from: {trained_model_path}")
    state_dict = torch.load(trained_model_path, map_location=torch.device('cpu'))
    loaded_quantized_model.load_state_dict(state_dict)
    print("✅ Successfully loaded trained weights.")
    
    # --- Step 2: Instantiate Placeholder Model ---
    print("\n--- Step 2: Instantiate Placeholder Model and Load Weights ---")
    placeholder_model = ITALSTMNetVIT_Placeholder_Export()
    placeholder_model.eval()

    # Copy the dequantized float weights (for Linear, Conv, etc.)
    placeholder_model = load_weights_to_placeholder_model(
        loaded_quantized_model,
        placeholder_model
    )
    
    # --- THIS IS THE CORRECTED PARAMETER COPYING LOGIC ---
    print("\n--- Step 3: Copying quantization parameters (scale/zero_point) ---")
    stub_lists_to_update = ["quant_attention", "quant_ffn"]

    for list_name in stub_lists_to_update:
        # Get the QuantStub modules from the loaded trained model
        source_stubs = getattr(loaded_quantized_model, list_name)
        # Get the HardcodedQuantizer modules from the placeholder model
        dest_quantizers = getattr(placeholder_model, list_name)

        for i in range(len(dest_quantizers)):
            # Get the learned scale and zero_point from the trained model's stub
            source_stub = source_stubs[i]
            scale = source_stub.qconfig.activation().scale.item()
            zero_point = source_stub.qconfig.activation().zero_point.item()
            
            # Get the corresponding quantizer in the placeholder model
            dest_quantizer = dest_quantizers[i]
            
            print(f"  - Updating {list_name}.{i} with scale={scale:.4f}, zp={zero_point}")

            # Update the buffers of the EXISTING module IN-PLACE.
            dest_quantizer.scale.copy_(torch.tensor(scale))
            dest_quantizer.zero_point.copy_(torch.tensor(zero_point))
            
    # Now the model is fully ready for export
    final_model_to_export = placeholder_model
    
    print("\n--- Step 4: Create Dummy Inputs for Tracing ---")
    batch_size = 1
    img_data = torch.randn(batch_size, 1, 60, 90)
    additional_data = torch.randn(batch_size, 1)
    quat_data = torch.randn(batch_size, 4)
    h_0 = torch.randn(3, batch_size, 128)
    c_0 = torch.randn(3, batch_size, 128)
    dummy_input = (img_data, additional_data, quat_data, h_0, c_0)

    output_path = "ita_placeholder_model_opset17.onnx"
    print(f"\n--- Step 5: Exporting Model to ONNX at '{output_path}' ---")

    torch.onnx.export(
        final_model_to_export,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=['image', 'additional_data', 'quat_data', 'hidden_in', 'cell_in'],
        output_names=['output_logits', 'hidden_out', 'cell_out'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'additional_data': {0: 'batch_size'},
            'quat_data': {0: 'batch_size'},
            'hidden_in': {1: 'batch_size'},
            'cell_in': {1: 'batch_size'},
            'output_logits': {0: 'batch_size'},
            'hidden_out': {1: 'batch_size'},
            'cell_out': {1: 'batch_size'},
        }
    )
    print(f"\n✅ Successfully exported model to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export placeholder model to ONNX.")
    parser.add_argument("model_path", type=str, help="Path to the trained .pth checkpoint.")
    args = parser.parse_args()
    export_placeholder_model(args.model_path)