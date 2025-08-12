# export_onnx.py

import torch
import os, sys

# Adjust these imports based on your project structure.
# This assumes the script is in the 'training' folder.

# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.ITA.QAT.model import ITALSTMNetVIT_QAT
from models.ITA.export.ITA_ONNX import ITAForONNXExport

QUANTIZED_ONNX_PATH = "ita_model_for_hardware.onnx"
MLIR_PATH = "ita_model_for_hardware.mlir"

def main():
    """
    Loads the trained QAT model, transfers weights to the export model,
    and exports the result to an ONNX file.
    """
    print("--- Starting ONNX Export Process ---")

    # --- Step 1: Load the trained and quantized model state ---
    
    # Path to the saved model from your QAT run
    trained_model_path = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_11_t16_00_qat_replace_model/model_quantized_final.pth"
    
    if not os.path.exists(trained_model_path):
        print(f"Error: Trained model not found at {trained_model_path}")
        return

    print(f"Loading trained quantized state_dict from: {trained_model_path}")
    
    # Instantiate a fresh QAT model on the CPU in eval mode
    # This model will act as the source for our float weights.
    trained_model = ITALSTMNetVIT_QAT()
    trained_model.eval()
    
    # Load the state dict from the .pth file
    trained_model.load_state_dict(torch.load(trained_model_path), strict=False)
    
    print("Successfully loaded trained model.")

    # --- Step 2: Instantiate the export model and transfer weights ---
    
    print("Instantiating the ONNX export model (ITAForONNXExport)...")
    model_for_export = ITAForONNXExport()
    model_for_export.eval()
    
    print("Transferring float weights from trained model to export model...")
    model_for_export.load_float_weights_from_trained_model(trained_model)
    
    # --- Step 3: Create a dummy input for tracing ---
    
    print("Creating dummy input for ONNX tracing...")
    # These shapes should match the expected input dimensions of your model
    batch_size = 1
    img_data = torch.randn(batch_size, 1, 60, 90, requires_grad=False)
    additional_data = torch.randn(batch_size, 1, requires_grad=False)
    quat_data = torch.randn(batch_size, 4, requires_grad=False)
    
    # LSTM hidden state: (num_layers, batch_size, hidden_size)
    hidden_state = (
        torch.randn(3, batch_size, 128, requires_grad=False),
        torch.randn(3, batch_size, 128, requires_grad=False)
    )
    
    dummy_input = [img_data, additional_data, quat_data, hidden_state]

    # --- Step 4: Export to ONNX ---
    
    output_onnx_path = "ita_model_for_hardware.onnx"
    print(f"Exporting model to {output_onnx_path}...")
    
    torch.onnx.export(
        model_for_export,
        (dummy_input,),
        output_onnx_path,
        export_params=True,
        opset_version=17,  # A reasonably modern opset version
        do_constant_folding=True,
        input_names=['image', 'additional_data', 'quat_data', 'hidden_in_h', 'hidden_in_c'],
        output_names=['output', 'hidden_out_h', 'hidden_out_c'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'additional_data': {0: 'batch_size'},
            'quat_data': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("--- ONNX Export Complete! --- ✅")
    print(f"Model saved to: {os.path.abspath(output_onnx_path)}")
    
    print("\n✅ Step 4: Exporting to MLIR...")
    print("To convert the quantized ONNX model to MLIR, run the following command in your terminal:")
    print("-" * 70)
    # Use IREE's (Intermediate Representation and Execution Environment) tool
    # You may need to add flags to target your specific hardware accelerator,
    # for example: --iree-hal-target-backends=...
    mlir_command = f"iree-import-onnx {QUANTIZED_ONNX_PATH} --opset-version 17 -o {MLIR_PATH} "
    print(f"👉 \033[1m{mlir_command}\033[0m")
    print("-" * 70)
    print("\nPTQ process with ONNX and MLIR export instructions are complete! 🎉")


if __name__ == "__main__":
    main()