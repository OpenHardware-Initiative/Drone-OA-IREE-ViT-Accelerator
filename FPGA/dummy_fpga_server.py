# ~/catkin_ws/src/openhardware-initiative-vitfly-fpga/envtest/ros/dummy_fpga_server.py
import socket
import numpy as np
import struct

# We need to import the functions and constants from the link file
# to ensure our dummy server speaks the same language as the client.
from fpga_link import unpack_frame, pack_reply, PORT

print("--- Dummy FPGA Server ---")
print("This script simulates the FPGA for a loopback test.")

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to listen on all available IPs on the specified port
server_address = ('0.0.0.0', PORT)
sock.bind(server_address)
print(f"Listening for packets on port {PORT}...")

while True:
    try:
        # 1. Wait and receive a packet from the client (the simulation)
        # The buffer size 8192 is large enough for our packet.
        packet, client_address = sock.recvfrom(8192)

        # 2. Unpack the packet using the protocol definition.
        # This verifies that the client is sending data in the correct format.
        img, des_vel, pos_x, quat = unpack_frame(packet)
        print(f"Received frame. Desired vel: {des_vel:.2f}, Pos X: {pos_x:.2f}, Img shape: {img.shape}")

        # 3. Define a simple, predictable dummy command.
        # We'll command the drone to fly straight forward at 2 m/s.
        # The format is [vx, vy, vz] in the world frame.
        dummy_velocity_cmd = np.array([2.0, 0.0, 0.0], dtype=np.float32)

        # 4. Pack the dummy reply using the protocol definition.
        reply_packet = pack_reply(dummy_velocity_cmd)

        # 5. Send the reply back to the client.
        sock.sendto(reply_packet, client_address)
        print(f"Sent dummy command {dummy_velocity_cmd} back to {client_address}")

    except Exception as e:
        print(f"An error occurred in the dummy server: {e}")
