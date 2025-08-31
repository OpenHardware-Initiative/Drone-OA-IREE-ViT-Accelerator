# export_to_onnx.py

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime
import sys
import argparse # For command-line arguments

# --- 1. Import the Model Definition ---
try:
    from models.ITA_single_layer_upsample_shuffle.model import ITALSTMNetVIT
except ImportError:
    print("Error: Could not import ITALSTMNetVIT.")
    print("Please make sure 'model.py' is in the same directory or in the Python path.")
    sys.exit(1)

# --- 2. Create a Wrapper for ONNX Export ---
# This wrapper remains the same.
class ITALSTMNetVIT_ONNXWrapper(nn.Module):
    def __init__(self, model: ITALSTMNetVIT):
        super().__init__()
        self.model = model

    def forward(self, img_data, additional_data, quat_data, h_in, c_in):
        hidden_state_in = (h_in, c_in)
        X = [img_data, additional_data, quat_data, hidden_state_in]
        out_final, hidden_state_out = self.model(X)
        h_out, c_out = hidden_state_out
        return out_final, h_out, c_out

# --- 3. Main Export and Verification Logic ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Load a PyTorch model from a .pth file and export it to ONNX."
    )
    parser.add_argument(
        "--input-weights",
        type=str,
        required=True,
        help="The path to the input .pth model weights file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The file path where the ONNX model will be saved."
    )
    args = parser.parse_args()
    
    # --- Configuration ---
    NUM_LAYERS = 1
    BATCH_SIZE = 1 
    LSTM_LAYERS = 3
    LSTM_HIDDEN = 128
    
    # --- Model Preparation ---
    print("Step 1: Preparing the PyTorch model...")
    # First, instantiate the model with the correct architecture
    base_model = ITALSTMNetVIT(num_layers=NUM_LAYERS)
    
    # Second, load the trained weights from the specified .pth file
    print(f"--> Loading weights from: '{args.input_weights}'")
    # Load onto CPU for maximum compatibility during export
    device = torch.device('cpu')
    try:
        # Load the state dictionary from the file
        state_dict = torch.load(args.input_weights, map_location=device)
        # Apply the loaded weights to the model instance
        base_model.load_state_dict(state_dict)
        print("--> Weights loaded successfully. ✅")
    except Exception as e:
        print(f"🚨 Error loading weights file: {e}")
        sys.exit(1)

    # Third, wrap the model for ONNX and set to evaluation mode
    model_for_export = ITALSTMNetVIT_ONNXWrapper(base_model)
    model_for_export.eval()
    print("✅ Model prepared for export.")

    # --- Create Dummy Inputs ---
    print(f"\nStep 2: Creating dummy inputs for tracing...")
    dummy_img = torch.randn(BATCH_SIZE, 1, 60, 90)
    dummy_add_data = torch.randn(BATCH_SIZE, 1)
    dummy_quat = torch.randn(BATCH_SIZE, 4)
    dummy_h_in = torch.randn(LSTM_LAYERS, BATCH_SIZE, LSTM_HIDDEN)
    dummy_c_in = torch.randn(LSTM_LAYERS, BATCH_SIZE, LSTM_HIDDEN)
    dummy_inputs = (dummy_img, dummy_add_data, dummy_quat, dummy_h_in, dummy_c_in)
    print("✅ Dummy inputs created.")

    # --- Export to ONNX ---
    print(f"\nStep 3: Exporting model to ONNX at '{args.output_path}'...")
    input_names = ["image_input", "additional_input", "quat_input", "h_in", "c_in"]
    output_names = ["final_output", "h_out", "c_out"]
    
    torch.onnx.export(
        model_for_export,
        dummy_inputs,
        args.output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=17
    )
    print(f"✅ Model successfully exported.")

    # --- Verification ---
    print("\nStep 4: Verifying the exported ONNX model...")
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    print("🔍 ONNX checker passed.")

    ort_session = onnxruntime.InferenceSession(args.output_path)
    ort_inputs = {name: tensor.numpy() for name, tensor in zip(input_names, dummy_inputs)}
    
    pytorch_outputs = model_for_export(*dummy_inputs)
    ort_outputs = ort_session.run(None, ort_inputs)
    
    for i, name in enumerate(output_names):
        pytorch_res = pytorch_outputs[i].detach().numpy()
        ort_res = ort_outputs[i]
        
        # Calculate and print the maximum absolute difference
        max_diff = np.max(np.abs(pytorch_res - ort_res))
        
        if np.allclose(pytorch_res, ort_res, atol=1e-5):
            print(f"  - Output '{name}' matches. 👍")
        else:
            # Print the detailed mismatch information
            print(f"  - 🚨 Output '{name}' MISMATCH! 🚨")
            print(f"    Max absolute difference: {max_diff}")


    print("\n✅ Verification complete.")

if __name__ == "__main__":
    main()