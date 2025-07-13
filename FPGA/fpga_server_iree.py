#!/usr/bin/python3
# fpga_server_iree.py

import socket
import numpy as np
import time
import sys
import os
import subprocess
import re # Import the regular expression module

from fpga_link import unpack_frame, pack_reply, PORT

def parse_iree_output_robust(cli_output_str):
    """
    Robustly parses the multi-line text output from the 'iree-run-module' tool.
    Handles multi-bracket format and checks for numerical issues like NAN.
    
    Args:
        cli_output_str (str): The stdout from the IREE command.
        
    Returns:
        A tuple containing:
        - raw_output (np.ndarray): The 1x3 velocity tensor.
        - new_hidden_state (tuple): A tuple of (new_h, new_c) numpy arrays.
    """
    try:
        # This pattern correctly finds all result blocks, even with complex formatting.
        pattern = r"result\[\d+\]:.*?=\s*(.*?)(?=\s*result\[\d+\]:|\Z)"
        matches = re.findall(pattern, cli_output_str, re.DOTALL)
        
        if not matches:
            raise ValueError("No result blocks found in the IREE output.")

        # --- Parse Result 0: Velocity Command ---
        # The key change is here: remove brackets entirely.
        vel_str = matches[0].replace('[', '').replace(']', '').replace('\n', ' ').strip()
        raw_output = np.fromstring(vel_str, dtype=np.float32, sep=' ')
        
        if raw_output.size != 3:
            raise ValueError(f"Expected 3 values for velocity, but parsed {raw_output.size}.")
        
        # Check for NANs or infinities, which indicate a model health problem.
        if not np.isfinite(raw_output).all():
            print("!!! WARNING: NAN or Infinity detected in velocity output! Using zeros instead. !!!")
            raw_output = np.zeros(3, dtype=np.float32)

        # --- Parse Results 1 and 2: Hidden States (h and c) ---
        new_hidden_state = (None, None)
        if len(matches) >= 3:
            # Also remove brackets entirely for hidden states.
            h_str = matches[1].replace('[', '').replace(']', '').replace('\n', ' ').strip()
            new_h = np.fromstring(h_str, dtype=np.float32, sep=' ')
            
            c_str = matches[2].replace('[', '').replace(']', '').replace('\n', ' ').strip()
            new_c = np.fromstring(c_str, dtype=np.float32, sep=' ')

            # Add size validation
            expected_size = 3 * 128
            if new_h.size != expected_size or new_c.size != expected_size:
                raise ValueError(f"Expected {expected_size} values for hidden state, but parsed h={new_h.size}, c={new_c.size}.")
            
            # Reshape after validation
            new_h = new_h.reshape(1, 3, 1, 128)
            new_c = new_c.reshape(1, 3, 1, 128)
            
            # If the hidden states are invalid, don't propagate them.
            if not np.isfinite(new_h).all() or not np.isfinite(new_c).all():
                print("!!! WARNING: NAN or Infinity detected in hidden states! Not propagating them. !!!")
                new_hidden_state = (None, None)
            else:
                new_hidden_state = (new_h, new_c)
        
        return raw_output, new_hidden_state

    except Exception as e:
        print("---!!! FAILED TO PARSE IREE CLI OUTPUT (Robust Parser v2) !!!---")
        print("Original output was:")
        print(f"'{cli_output_str}'")
        raise e # Re-raise the exception to be caught by the main loop

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
    print("--- IREE CLI-based Inference Server ---")

    iree_tool_path = "/home/ubuntu/iree-cortex-a53/bin/iree-run-module"
    model_path = "/home/ubuntu/FPGA/models/lstmnetvit_f16_optimized_debug.vmfb"
    
    temp_dir = "/tmp/iree_io"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using temporary directory for I/O: {temp_dir}")
    
    h_path = os.path.join(temp_dir, "h_state.npy")
    c_path = os.path.join(temp_dir, "c_state.npy")
    np.save(h_path, np.zeros((1, 3, 1, 128), dtype=np.float32))
    np.save(c_path, np.zeros((1, 3, 1, 128), dtype=np.float32))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('0.0.0.0', PORT)
    sock.bind(server_address)
    print(f"Network server running. Listening on port {PORT}...")
    print("-----------------------------------------")

    while True:
        try:
            packet, client_address = sock.recvfrom(8192)
            img_u8, desired_vel, pos_x, quat = unpack_frame(packet)
            
            img_f32 = (img_u8.astype(np.float32) / 255.0)
            img_path = os.path.join(temp_dir, "input_img.npy")
            np.save(img_path, img_f32)

            command = [
                iree_tool_path,
                "--device=local-sync",
                f"--module={model_path}",
                f"--function=main_graph",
                f"--input=1x1x60x90xf32=@{img_path}",
                f"--input=1x1xf32={desired_vel}",
                f"--input=1x4xf32=[{quat[0]},{quat[1]},{quat[2]},{quat[3]}]",
                f"--input=1x3x1x128xf32=@{h_path}",
                f"--input=1x3x1x128xf32=@{c_path}",
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("---!!! IREE CLI tool failed! !!!---")
                print("Command that failed:", " ".join(command))
                print("Return Code:", result.returncode)
                print("Stderr:", result.stderr)
                continue
                
            raw_output, new_hidden_state = parse_iree_output_robust(result.stdout)
            
            # This check is important: only save the new state if it's valid.
            if new_hidden_state[0] is not None and new_hidden_state[1] is not None:
                np.save(h_path, new_hidden_state[0])
                np.save(c_path, new_hidden_state[1])
            
            final_velocity_cmd = calculate_final_velocity(raw_output, desired_vel, pos_x)
            reply_packet = pack_reply(final_velocity_cmd)
            sock.sendto(reply_packet, client_address)

        except Exception as e:
            print(f"An error occurred in the server loop: {type(e).__name__}: {e}")
            print("Parsing failed or another error occurred, skipping this frame.")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()