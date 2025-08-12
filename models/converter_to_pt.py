#!/usr/bin/python3
import torch
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# 1. IMPORT YOUR MODEL'S CLASS DEFINITION
# ==========================================
# IMPORTANT: Change 'your_model_file' to the name of the Python file
# where your LSTMNetVIT class is defined.
from third_party.vitfly_FPGA.models.model import LSTMNetVIT

# --- Configuration ---
PTH_WEIGHTS_PATH = "ViTLSTM_model.pth"
OUTPUT_PT_PATH = "model_deployable.pt"

def convert_pth_to_torchscript():
    """
    Loads your LSTMNetVIT architecture, applies weights from the .pth file,
    and traces it to produce a deployable .pt TorchScript file.
    """
    print("Starting model conversion process...")

    # Step 1: Instantiate your model architecture.
    # (This assumes the __init__ method takes no arguments, which is true for your class)
    try:
        model = LSTMNetVIT()
        print("✅ Step 1: LSTMNetVIT architecture instantiated.")
    except NameError:
        print("\n[ERROR] The model class 'LSTMNetVIT' was not found.")
        print("        Please edit line 8 to correctly import your model's Python class.\n")
        return

    # Step 2: Load the saved weights from your .pth file.
    if not os.path.exists(PTH_WEIGHTS_PATH):
        print(f"\n[ERROR] Weights file not found at: {PTH_WEIGHTS_PATH}\n")
        return
        
    state_dict = torch.load(PTH_WEIGHTS_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print(f"✅ Step 2: Weights loaded from '{PTH_WEIGHTS_PATH}'.")

    # Step 3: Set the model to evaluation mode.
    model.eval()
    print("✅ Step 3: Model set to evaluation mode.")

    # Step 4: Create dummy inputs that EXACTLY MATCH your inference code.
    print("✅ Step 4: Creating dummy inputs based on your inference script...")
    
    # Input shapes based on compute_command_vision_based:
    h, w = (60, 90)
    img_input = torch.rand(1, 1, h, w, dtype=torch.float32)
    vel_input = torch.rand(1, 1, dtype=torch.float32)
    quat_input = torch.rand(1, 4, dtype=torch.float32)
    
    # The LSTM hidden state is a tuple of (h_0, c_0)
    # The shapes are (num_layers, hidden_size) from your code, but for tracing a
    # batched model, we need (num_layers, batch_size, hidden_size).
    num_layers = 3
    hidden_size = 128
    batch_size = 1 # We are tracing a single inference.
    h0_input = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32)
    c0_input = torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32)
    hidden_state_tuple = (h0_input, c0_input)
    
    # The model's forward() method expects a single argument: a LIST containing these tensors.
    example_input_list = [img_input, vel_input, quat_input, hidden_state_tuple]
    print("    - Input structure created: [image, velocity, quaternion, (h_0, c_0)]")

    # Step 5: Trace the model.
    print("Step 5: Tracing the model...")
    try:
        # We pass a tuple containing our single list argument to the tracer.
        traced_module = torch.jit.trace(model, (example_input_list,))
        print("✅ Step 5: Model traced successfully.")
    except Exception as e:
        print(f"\n[ERROR] Tracing failed. The model's forward pass raised an error.")
        print("        This can happen if there is a shape mismatch inside your model's layers.")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Save the traced model to the output .pt file.
    traced_module.save(OUTPUT_PT_PATH)
    
    print("-" * 60)
    print(f"✅ SUCCESS! Model converted and saved to '{OUTPUT_PT_PATH}'")
    print("\nYou can now use this file with IREE.")
    print("-" * 60)

if __name__ == '__main__':
    convert_pth_to_torchscript()