# fpga_link.py
import struct, time, itertools
import numpy as np
import torch

SOF      = 0xA55A1234
PORT     = 5555
FRAME_H  = 60
FRAME_W  = 90
HIDDEN_SHAPE = (3, 128)          # matches LSTMNetVIT
SEQ_GEN  = itertools.count()     # global monotonous counter

# ---------- pack / unpack ------------------------------------------------ #
def pack_frame(depth_u8, des_vel, quat, hidden_state):
    """
    depth_u8      : np.uint8 array of shape (60, 90)
    des_vel       : float
    quat          : iterable of 4 floats  (w, x, y, z)
    hidden_state  : (h, c) tuple of torch Tensors  shape (3,128)
    returns bytes
    """
    h, c = (_to_fp16(x) for x in hidden_state)
    header = struct.pack(
        "<IIdf4f", SOF, next(SEQ_GEN), time.time(), des_vel, *quat
    )
    return b"".join([header,
                     depth_u8.tobytes(order="C"),
                     h.tobytes(order="C"),
                     c.tobytes(order="C")])

def unpack_reply(buf):
    """
    buf -- raw bytes sent back by server
    layout  : 3 × float32  + 2 × hidden_state(fp16)
    returns : vel (np.float32[3]),  (h, c) torch.float32
    """
    vel = np.frombuffer(buf[:12], dtype=np.float32)
    h   = np.frombuffer(buf[12:12+HIDDEN_SHAPE[0]*HIDDEN_SHAPE[1]*2],
                        dtype=np.float16).astype(np.float32).reshape(HIDDEN_SHAPE)
    c   = np.frombuffer(buf[12+len(h.tobytes()):],
                        dtype=np.float16).astype(np.float32).reshape(HIDDEN_SHAPE)
    return vel, (torch.from_numpy(h), torch.from_numpy(c))

# ---------- helpers ------------------------------------------------------ #
def empty_hidden(device="cpu"):
    z = torch.zeros(*HIDDEN_SHAPE, dtype=torch.float32, device=device)
    return (z.clone(), z.clone())

def _to_fp16(t):
    if t is None:
        t = torch.zeros(*HIDDEN_SHAPE)
    return t.cpu().numpy().astype(np.float16, copy=False)
