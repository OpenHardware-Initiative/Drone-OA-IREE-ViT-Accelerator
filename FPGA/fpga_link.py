# envtest/ros/fpga_link.py (Version 2 - Stateful Server)

import struct
import numpy as np

# Network Configuration
PORT = 10001
FPGA_IP = "10.42.0.14" # IMPORTANT: Change this to your FPGA's IP address

# Data format for HOST -> FPGA (No hidden state)
# - Image: 60x90 uint8 = 5400 bytes
# - Desired Velocity: float32 = 4 bytes
# - Position X: float32 = 4 bytes
# - Quaternion: 4x float32 = 16 bytes
PACKET_FMT_SEND = '>5400sff16s' # Note: Quat is now 16s

# Data format for FPGA -> HOST (No hidden state)
# - Velocity Command: 3x float32 = 12 bytes
PACKET_FMT_REPLY = '>12s'

def pack_frame(depth_u8, desired_vel, pos_x, quat):
    """Packs data for sending to the FPGA. (No hidden state)"""
    img_bytes = depth_u8.tobytes()
    quat_bytes = quat.astype(np.float32).tobytes()

    assert len(img_bytes) == 5400
    assert len(quat_bytes) == 16

    return struct.pack(PACKET_FMT_SEND, img_bytes, desired_vel, pos_x, quat_bytes)

def unpack_frame(packet):
    """Unpacks data received on the FPGA. (No hidden state)"""
    img_bytes, desired_vel, pos_x, quat_bytes = struct.unpack(PACKET_FMT_SEND, packet)

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(60, 90).copy()
    quat = np.frombuffer(quat_bytes, dtype=np.float32)

    return img, desired_vel, pos_x, quat


def pack_reply(velocity_cmd):
    """Packs the reply data for sending back to the host. (No hidden state)"""
    vel_bytes = velocity_cmd.astype(np.float32).tobytes()
    assert len(vel_bytes) == 12
    return struct.pack(PACKET_FMT_REPLY, vel_bytes)


def unpack_reply(packet):
    """Unpacks the reply data received on the host. (No hidden state)"""
    (vel_bytes,) = struct.unpack(PACKET_FMT_REPLY, packet) # Note tuple unpacking
    velocity = np.frombuffer(vel_bytes, dtype=np.float32)
    return velocity
