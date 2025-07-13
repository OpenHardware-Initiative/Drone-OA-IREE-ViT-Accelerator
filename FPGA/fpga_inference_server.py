# fpga_inference_server.py (Modified to be a callable module)

import numpy as np
import torch
from torchvision.transforms import ToTensor
from models.model import LSTMNetVIT # Path will be fixed with sys.path

# This function is now standalone, but still part of this file
def calculate_final_velocity(raw_output, desired_vel, pos_x):
    vel_cmd = raw_output.copy()
    vel_cmd[0] = np.clip(vel_cmd[0], -1, 1)
    norm = np.linalg.norm(vel_cmd)
    if norm > 0:
        vel_cmd = vel_cmd / norm
    final_velocity = vel_cmd * desired_vel

    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if pos_x < hardcoded_ctl_threshold:
        final_velocity[0] = max(min_xvel_cmd, (pos_x / hardcoded_ctl_threshold) * desired_vel)
    return final_velocity

def initialize_model(model_path):
    """Loads the model and creates the initial hidden state."""
    print(f"Loading model from {model_path}...")
    device = torch.device("cpu")
    model = LSTMNetVIT().to(device).float()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    hidden_state = (
        torch.zeros(3, 128, device=device),
        torch.zeros(3, 128, device=device)
    )
    print("Internal hidden state initialized.")
    return model, hidden_state

def run_inference_step(model, hidden_state, img_u8, desired_vel, quat):
    """Runs one step of inference and returns the raw output and new state."""
    device = next(model.parameters()).device
    
    # Prepare tensors
    img_tensor = ToTensor()(img_u8).view(1, 1, 60, 90).float().to(device)
    vel_tensor = torch.tensor(desired_vel).view(1, 1).float().to(device)
    quat_tensor = torch.tensor(quat).view(1, -1).float().to(device)

    # Run Inference
    with torch.no_grad():
        raw_output_tensor, new_hidden_state = model([img_tensor, vel_tensor, quat_tensor, hidden_state])

    raw_output_np = raw_output_tensor.squeeze().detach().cpu().numpy()
    
    return raw_output_np, new_hidden_state