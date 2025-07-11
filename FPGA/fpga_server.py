#!/usr/bin/python3
# fpga_server.py

import socket
import numpy as np
import torch

# --- CORRECTED LOCAL IMPORTS ---
# These files are now in the same directory, so we import directly.
from fpga_link import unpack_frame, pack_reply, PORT
from fpga_inference_server import initialize_model, run_inference_step, calculate_final_velocity

def main():
    print("--- ViT+LSTM FPGA Inference Server ---")

    # --- 1. Initialize the Model ---
    # The weights file should be in a predictable location relative to this script.
    model_path = "./models/ViTLSTM_model.pth" # This path is now relative to this file's location.
    model, hidden_state = initialize_model(model_path)

    # --- 2. Setup Network Server ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('0.0.0.0', PORT)
    sock.bind(server_address)
    print(f"Network server is listening for packets on port {PORT}...")

    # --- 3. Main Network and Inference Loop ---
    while True:
        try:
            print("\n[SERVER] Waiting for packet...")
            packet, client_address = sock.recvfrom(8192)
            print(f"[SERVER] Received {len(packet)} bytes from {client_address}")
            print("[SERVER] Unpacking frame...")
            img_u8, desired_vel, pos_x, quat = unpack_frame(packet)
            print("[SERVER] Unpack complete. Image shape:", img_u8.shape)
            img_u8 = img_u8.copy()
            print("[DEBUG] Data unpacked. Calling inference engine...")
            raw_output, new_hidden_state = run_inference_step(model, hidden_state, img_u8, desired_vel, quat)
            hidden_state = new_hidden_state
            print("[DEBUG] Inference complete. Sending reply...")
            final_velocity_cmd = calculate_final_velocity(raw_output, desired_vel, pos_x)
            reply_packet = pack_reply(final_velocity_cmd)
            sock.sendto(reply_packet, client_address)

        except Exception as e:
            print(f"An error occurred in the server loop: {e}")
            import traceback
            traceback.print_exc()
            hidden_state = (
                torch.zeros(3, 128, device=next(model.parameters()).device),
                torch.zeros(3, 128, device=next(model.parameters()).device)
            )
            print("Internal hidden state has been reset due to an error.")

if __name__ == "__main__":
    main()