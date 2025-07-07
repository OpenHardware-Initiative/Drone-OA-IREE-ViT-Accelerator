# test_fpga_server.py
import time
import numpy as np
import torch
from torchvision.transforms import ToTensor

# --- Local Imports on FPGA ---
# Make sure these files are in the same directory or in Python's path on the FPGA
from model import LSTMNetVIT

def calculate_final_velocity(raw_output, desired_vel, pos_x):
    """
    This function contains the post-processing logic that was previously on the host.
    """
    # Normalize the raw output
    vel_cmd = raw_output
    vel_cmd[0] = np.clip(vel_cmd[0], -1, 1)
    # Add a small epsilon to avoid division by zero if the norm is zero
    norm = np.linalg.norm(vel_cmd)
    if norm == 0:
        print("Warning: Zero norm for velocity command.")
        return np.zeros_like(vel_cmd)
        
    vel_cmd = vel_cmd / norm
    final_velocity = vel_cmd * desired_vel

    # Apply manual speedup logic
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if pos_x < hardcoded_ctl_threshold:
        final_velocity[0] = max(min_xvel_cmd, (pos_x / hardcoded_ctl_threshold) * desired_vel)

    return final_velocity

def run_test():
    """
    Main test function to run a single inference cycle with dummy data.
    """
    print("--- Starting FPGA Inference Test ---")

    # --- 1. Load Model ---
    # IMPORTANT: Update this path to where your model is stored on the FPGA
    model_path = "ViTLSTM_model.pth"
    try:
        print(f"Loading model from {model_path}...")
        device = torch.device("cpu")  # Assuming CPU on FPGA
        model = LSTMNetVIT().to(device).float()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("[SUCCESS] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # --- 2. Generate Dummy Input Data ---
    print("\nGenerating dummy input data...")
    # Dummy image (60x90, uint8), e.g., a simple gradient
    img_u8 = np.array([np.arange(90) for _ in range(60)], dtype=np.uint8)
    # Dummy state values
    desired_vel = 5.0
    pos_x = 1.5
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # No rotation
    # Dummy hidden state (zeros) for the LSTM
    hidden_state = (
        torch.zeros(3, 128, device=device), # h_0 for ViTLSTM
        torch.zeros(3, 128, device=device)  # c_0 for ViTLSTM
    )
    print("[SUCCESS] Dummy data generated.")
    print(f"  - Desired Velocity: {desired_vel}, Position X: {pos_x}")


    # --- 3. Prepare Tensors for Inference ---
    print("\nPreparing tensors for inference...")
    try:
        img_tensor = ToTensor()(img_u8).view(1, 1, 60, 90).float().to(device)
        vel_tensor = torch.tensor(desired_vel).view(1, 1).float().to(device)
        quat_tensor = torch.tensor(quat).view(1, -1).float().to(device)
        h_in = hidden_state
        print("[SUCCESS] Tensors prepared.")
        print(f"  - Image Tensor Shape: {img_tensor.shape}")
        print(f"  - Hidden State Shape: ({h_in[0].shape}, {h_in[1].shape})")
    except Exception as e:
        print(f"[ERROR] Failed to prepare tensors: {e}")
        return

    # --- 4. Run Inference ---
    print("\nRunning model inference...")
    start_time = time.time()
    try:
        with torch.no_grad():
            raw_output_tensor, h_out = model([img_tensor, vel_tensor, quat_tensor, h_in])
        end_time = time.time()
        print(f"[SUCCESS] Inference completed in {end_time - start_time:.4f} seconds.")
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return

    # --- 5. Post-process to Get Final Command ---
    print("\nCalculating final velocity command...")
    raw_output_np = raw_output_tensor.squeeze().cpu().numpy()
    final_velocity_cmd = calculate_final_velocity(raw_output_np, desired_vel, pos_x)
    print("[SUCCESS] Post-processing complete.")

    # --- 6. Display Results ---
    print("\n--- TEST RESULTS ---")
    print(f"Raw Model Output (numpy): {raw_output_np}")
    print(f"Final Velocity Command: {final_velocity_cmd}")
    print(f"New Hidden State Shapes: ({h_out[0].shape}, {h_out[1].shape})")
    print("--- Test Finished ---")


if __name__ == "__main__":
    run_test()