# fpga_inference_server.py (Version 2 - Stateful)

import socket
import numpy as np
import torch
from torchvision.transforms import ToTensor

from model import LSTMNetVIT
from fpga_link import unpack_frame, pack_reply, PORT

def calculate_final_velocity(raw_output, desired_vel, pos_x):
    # This function remains the same
    vel_cmd = raw_output
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

def main():
    # --- Load Model ---
    model_path = "/path/to/your/ViTLSTM_model.pth" # IMPORTANT: Update this path
    print(f"Loading model from {model_path}...")
    device = torch.device("cpu")
    model = LSTMNetVIT().to(device).float()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Stateful Variable ---
    # The hidden state is now managed entirely by the server.
    # It's initialized once to zeros.
    hidden_state = (
        torch.zeros(3, 128, device=device),
        torch.zeros(3, 128, device=device)
    )
    print("Internal hidden state initialized.")

    # --- Setup Network ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', PORT))
    print(f"Listening for packets on port {PORT}...")

    # --- Main Loop ---
    while True:
        try:
            packet, addr = sock.recvfrom(8192)

            # 1. Unpack data (no hidden state received)
            img_u8, desired_vel, pos_x, quat = unpack_frame(packet)

            # 2. Prepare tensors
            img_tensor = ToTensor()(img_u8).view(1, 1, 60, 90).float().to(device)
            vel_tensor = torch.tensor(desired_vel).view(1, 1).float().to(device)
            quat_tensor = torch.tensor(quat).view(1, -1).float().to(device)

            # 3. Run Inference using the *internal* hidden_state
            with torch.no_grad():
                raw_output_tensor, h_out = model([img_tensor, vel_tensor, quat_tensor, hidden_state])

            # 4. IMPORTANT: Update the internal hidden_state for the next cycle
            hidden_state = h_out

            # 5. Post-process
            raw_output_np = raw_output_tensor.squeeze().detach().cpu().numpy()
            final_velocity_cmd = calculate_final_velocity(raw_output_np, desired_vel, pos_x)

            # 6. Pack and send reply (no hidden state sent)
            reply_packet = pack_reply(final_velocity_cmd)
            sock.sendto(reply_packet, addr)

        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally reset state on error
            hidden_state = (torch.zeros(3, 128, device=device), torch.zeros(3, 128, device=device))
            print("Internal hidden state has been reset due to an error.")

if __name__ == "__main__":
    main()