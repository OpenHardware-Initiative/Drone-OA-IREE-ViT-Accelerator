#!/usr/bin/python3
# fpga_server.py

import socket
import numpy as np
import time
import sys
import os
import subprocess 

from fpga_link import unpack_frame, pack_reply, PORT

def parse_iree_output(cli_output_str):
    """
    A more robust parser for the text output of iree-run-module.
    It finds result blocks and extracts the numpy arrays.
    """
    try:
        # Find all occurrences of "result[...]" to identify the start of each output block
        result_indices = [i for i, s in enumerate(cli_output_str.split('\n')) if 'result[' in s]
        
        # --- Parse Velocity Output (result[0]) ---
        start_line_idx = result_indices[0] + 1 # The data starts on the line AFTER "result[0]:..."
        # Combine all lines from the start of the data until the next result block or the end
        end_line_idx = result_indices[1] if len(result_indices) > 1 else len(cli_output_str.split('\n'))
        
        # Join all lines belonging to this result and remove brackets/newlines
        vel_str = "".join(cli_output_str.split('\n')[start_line_idx:end_line_idx])
        vel_str = vel_str.replace('[', '').replace(']', '').strip()
        raw_output = np.fromstring(vel_str, dtype=np.float32, sep=' ')

        # --- Parse Hidden States (if they exist) ---
        if len(result_indices) > 2:
            # Parse h-state (result[1])
            start_line_idx = result_indices[1] + 1
            end_line_idx = result_indices[2]
            h_str = "".join(cli_output_str.split('\n')[start_line_idx:end_line_idx])
            h_str = h_str.replace('[', '').replace(']', '').strip()
            new_h = np.fromstring(h_str, dtype=np.float32, sep=' ').reshape(1, 3, 1, 128)

            # Parse c-state (result[2])
            start_line_idx = result_indices[2] + 1
            end_line_idx = len(cli_output_str.split('\n'))
            c_str = "".join(cli_output_str.split('\n')[start_line_idx:end_line_idx])
            c_str = c_str.replace('[', '').replace(']', '').strip()
            new_c = np.fromstring(c_str, dtype=np.float32, sep=' ').reshape(1, 3, 1, 128)

            new_hidden_state = (new_h, new_c)
        else:
            # Handle case where model doesn't output hidden states
            new_hidden_state = (None, None)

        return raw_output, new_hidden_state

    except Exception as e:
        print("---!!! FAILED TO PARSE IREE CLI OUTPUT !!!---")
        print("Original output was:")
        print(cli_output_str)
        # We don't re-raise the exception, just return None to be handled by the main loop
        return None, (None, None)

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

    # --- Paths to our tools and files ---
    iree_tool_path = "/home/ubuntu/iree-cortex-a53/bin/iree-run-module"
    model_path = "/home/ubuntu/FPGA/models/lstmnetvit_f16_optimized_debug.vmfb"
    
    # Create a temporary directory for file I/O
    temp_dir = "/tmp/iree_io"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using temporary directory for I/O: {temp_dir}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('0.0.0.0', PORT)
    sock.bind(server_address)
    print(f"Network server running. Listening on port {PORT}...")
    print("-----------------------------------------")

    # --- 3. Main Network and Inference Loop ---
    while True:
        try:
            
            packet, client_address = sock.recvfrom(8192)
            img_u8, desired_vel, pos_x, quat = unpack_frame(packet)
            
            # 1. Save input image to a temporary file
            # Note: IREE's file input expects float32, not uint8
            img_f32 = (img_u8.astype(np.float32) / 255.0)
            img_path = os.path.join(temp_dir, "input_img.npy")
            np.save(img_path, img_f32)

            # 2. Construct the command-line arguments
            # We use @filepath to tell IREE to load the input from a file
            command = [
                iree_tool_path,
                "--device=local-sync",
                f"--module={model_path}",
                f"--function=main_graph",
                f"--input=1x1x60x90xf32=@{img_path}",
                f"--input=1x1xf32={desired_vel}", # Small inputs can be passed directly
                f"--input=1x4xf32=[{quat[0]},{quat[1]},{quat[2]},{quat[3]}]"
            ]
            
            # 3. Execute the command and capture output
            inference_start_time = time.time()
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            inference_end_time = time.time()
            
            # 4. Parse the text output to get the raw velocity
            # Note: This simple model version doesn't have hidden state I/O
            parsed_data = parse_iree_output(result.stdout)
            if parsed_data[0] is None:
                print("Parsing failed, skipping this frame.")
                # We 'continue' to the next iteration of the while loop
                # and reuse the old hidden state.
                continue 

            raw_output, new_hidden_state = parsed_data

            # Overwrite the hidden state files if the new state is valid
            if new_hidden_state[0] is not None:
                np.save(h_path, new_hidden_state[0])
                np.save(c_path, new_hidden_state[1])
            
            # 5. Post-process, pack, and send reply
            final_velocity_cmd = calculate_final_velocity(raw_output, desired_vel, pos_x)
            reply_packet = pack_reply(final_velocity_cmd)
            sock.sendto(reply_packet, client_address)

            # 6. Log performance
            inference_duration_ms = (inference_end_time - inference_start_time) * 1000
            vx, vy, vz = final_velocity_cmd
            print(f"   {inference_duration_ms:7.2f} ms   |   Vx: {vx:6.2f}, Vy: {vy:6.2f}, Vz: {vz:6.2f}")

        except subprocess.CalledProcessError as e:
            print("---!!! IREE CLI tool failed! !!!---")
            print("Command that failed:", " ".join(e.args))
            print("Return Code:", e.returncode)
            print("Stderr:", e.stderr)
        except Exception as e:
            print(f"An error occurred in the server loop: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()