#!/usr/bin/python3
# fpga_server.py

import socket
import numpy as np
import time
import sys
sys.path.append("/home/ubuntu/iree-cortex-a53/bindings/python")

try:
    # Using insert(0, ...) is more robust than append(...)
    # It puts our custom path at the front of the search list.
    sys.path.insert(0, "/home/ubuntu/iree-cortex-a53/bindings/python")
    from iree import runtime as ireert
except ImportError as e:
    print("--- FATAL ERROR: Could not import the IREE runtime ---")
    print(f"Please check that the path is correct: {e}")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)
from fpga_link import unpack_frame, pack_reply, PORT


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


def main():
    print("--- ViT+LSTM FPGA Inference Server ---")

   # --- 1. Initialize the IREE Model ---
    model_path = "./models/lstmnetvit_f16.vmfb" 
    print(f"Loading IREE compiled model from {model_path}...")

    config = ireert.Config("local-task")
    with open(model_path, "rb") as f:
        vm_module = ireert.VmModule.from_flatbuffer(f.read())

    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)

    # Use the correct function name from your colleague's command
    model_func = ctx.modules.module["main_graph"]
    print("IREE module loaded successfully.")

    # The hidden state is now two NumPy arrays
    hidden_state = (
        np.zeros((1, 3, 1, 128), dtype=np.float32), # h_0
        np.zeros((1, 3, 1, 128), dtype=np.float32)  # c_0
    )
    print("NumPy hidden state for IREE initialized.")

    # --- 2. Setup Network Server ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('0.0.0.0', PORT)
    sock.bind(server_address)
    print(f"Network server is running. Listening for packets on port {PORT}...")
    print("-----------------------------------------")
    print("-----------------------------------------")
    print("  Inference Time  |      Velocity Command (X, Y, Z)")
    print("-----------------------------------------")

    log_interval = 1.0  # Log data once per second
    last_log_time = time.time()
    inference_times = []

    # --- 3. Main Network and Inference Loop ---
    while True:
        try:
            
            packet, client_address = sock.recvfrom(8192)
            
            img_u8, desired_vel, pos_x, quat = unpack_frame(packet)
            img_u8 = img_u8.copy()
            
            inference_start_time = time.time()

            # a. Pre-process inputs into NumPy arrays of the correct shape/type
            img_f32 = (img_u8.astype(np.float32) / 255.0).reshape(1, 1, 60, 90)
            vel_f32 = np.array([[desired_vel]], dtype=np.float32)
            quat_f32 = quat.astype(np.float32).reshape(1, 4)
            h_in, c_in = hidden_state

            # b. Call the compiled IREE function
            raw_output_list, new_h, new_c = model_func(img_f32, vel_f32, quat_f32, h_in, c_in)

            inference_end_time = time.time()

            # c. Update the persistent hidden state with the NumPy outputs
            hidden_state = (new_h, new_c)

            # d. Get the raw output from the returned list
            raw_output = raw_output_list[0]
            
            final_velocity_cmd = calculate_final_velocity(raw_output, desired_vel, pos_x)
            reply_packet = pack_reply(final_velocity_cmd)
            sock.sendto(reply_packet, client_address)

            inference_duration_ms = (inference_end_time - inference_start_time) * 1000
            vx, vy, vz = final_velocity_cmd
            print(f"   {inference_duration_ms:7.2f} ms   |   Vx: {vx:6.2f}, Vy: {vy:6.2f}, Vz: {vz:6.2f}")

        except Exception as e:
            print(f"An error occurred in the server loop: {e}")
            import traceback
            traceback.print_exc()
            hidden_state = (
                np.zeros((1, 3, 1, 128), dtype=np.float32),
                np.zeros((1, 3, 1, 128), dtype=np.float32)
            )
            print("Internal hidden state has been reset due to an error.")

if __name__ == "__main__":
    main()