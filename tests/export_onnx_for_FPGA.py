# export_onnx.py

import torch
import os, sys

# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.ITA_single_layer_upsample_shuffle.model import ITALSTMNetVIT
from models.ITA_single_layer_upsample_shuffle.export.model import ITAForONNXExport

# --- Path Definitions (Corrected Naming) ---
FLOAT_ONNX_PATH = f"{PROJECT_ROOT}/models/ITA_single_layer_upsample_shuffle/export/ITAViTLSTM_float.onnx"
MLIR_PATH = f"{PROJECT_ROOT}/models/ITA_single_layer_upsample_shuffle/export/ITAViTLSTM_float.mlir"

def main():
    """
    Loads the trained QAT model, transfers weights to the export model,
    and exports the result to an ONNX file.
    """
    print("--- Starting ONNX Export Process ---")

    # --- Step 1: Load the trained and quantized model state ---
    
    trained_model_path = f"{PROJECT_ROOT}/models/ITA_single_layer_upsample_shuffle/model_000205.pth"
    print(f"Loading trained model state from: {trained_model_path}")
    if not os.path.exists(trained_model_path):
        print(f"Error: Trained model not found at {trained_model_path}")
        return

    # Instantiate a fresh QAT model on the CPU in eval mode
    trained_model = ITALSTMNetVIT()
    trained_model.eval()
    
    # ✅ CRITICAL FIX: Load state dict with map_location to ensure it works on CPU-only machines.
    state_dict = torch.load(trained_model_path, map_location=torch.device('cpu'))
    trained_model.load_state_dict(state_dict, strict=False)
    
    print("Successfully loaded trained model state onto CPU.")

    # --- Step 2: Instantiate the export model and transfer weights ---
    
    print("Instantiating the ONNX export model (ITAForONNXExport)...")
    model_for_export = ITAForONNXExport()
    model_for_export.eval()
    
    print("Transferring float weights from trained model to export model...")
    model_for_export.load_float_weights_from_trained_model(trained_model)
    
    # --- Step 3: Create a dummy input for tracing ---
    
    print("Creating dummy input for ONNX tracing...")
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
    
    print(f"Exporting model to {FLOAT_ONNX_PATH}...")
    
    torch.onnx.export(
        model_for_export,
        (dummy_input,),  # The comma is important to make it a tuple of arguments
        FLOAT_ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image', 'additional_data', 'quat_data', 'hidden_in_h', 'hidden_in_c'],
        output_names=['output', 'hidden_out_h', 'hidden_out_c'],
    )
    
    print("--- ONNX Export Complete! --- ✅")
    print(f"Model saved to: {os.path.abspath(FLOAT_ONNX_PATH)}")
    
    print("\n✅ Step 5: Exporting to MLIR...")
    print("To convert the ONNX model to MLIR, run the following command:")
    print("-" * 70)
    mlir_command = f"iree-import-onnx {FLOAT_ONNX_PATH} --opset-version 17 -o {MLIR_PATH}"
    print(f"👉 \033[1m{mlir_command}\033[0m")
    print("-" * 70)
    print("\nExport process instructions are complete! 🎉")


if __name__ == "__main__":
    main()