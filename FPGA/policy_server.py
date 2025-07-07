#!/usr/bin/env python3
import argparse, socket, sys, os, struct
from os.path import join as opj
import torch
import numpy as np
from fpga_link import unpack_reply, pack_frame, SOF, FRAME_H, FRAME_W, HIDDEN_SHAPE


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')))

from model import LSTMNetVIT

# same header format as pack_frame: "<IIdf4f>"
HDR_FMT = "<IIdf4f"
HDR_SIZE = struct.calcsize(HDR_FMT)

def main():
    # ─── argument parsing ──────────────────────────────────────────────
    p = argparse.ArgumentParser(description="UDP policy server for ViT+LSTM")
    p.add_argument("--model_path", required=True,
                   help="path to ViTLSTM_model.pth")
    p.add_argument("--port", type=int, default=5555,
                   help="UDP port to listen on")
    args = p.parse_args()

    # ─── load model ───────────────────────────────────────────────────
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNetVIT().to(dev).eval()
    model.load_state_dict(torch.load(args.model_path, map_location=dev))

    # ─── open socket ──────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    print(f"[policy-server] listening on UDP {args.port}")

    while True:
        pkt, addr = sock.recvfrom(16_000)
        # 1) header
        if len(pkt) < HDR_SIZE:
            continue
        sof, seq, tstamp, des_vel, qw, qx, qy, qz = struct.unpack(
            HDR_FMT, pkt[:HDR_SIZE]
        )
        if sof != SOF:
            continue

        # 2) depth
        offs = HDR_SIZE
        img_n = FRAME_H * FRAME_W
        depth_u8 = np.frombuffer(pkt[offs:offs+img_n], dtype=np.uint8)
        depth = (depth_u8.astype(np.float32) / 255.0)\
                    .reshape(1,1,FRAME_H,FRAME_W)
        offs += img_n

        # 3) hidden h, c in float16
        count_fp16 = np.prod(HIDDEN_SHAPE)
        h = np.frombuffer(pkt[offs:offs+2*count_fp16], dtype=np.float16)\
                .astype(np.float32).reshape(HIDDEN_SHAPE)
        offs += 2 * count_fp16
        c = np.frombuffer(pkt[offs:offs+2*count_fp16], dtype=np.float16)\
                .astype(np.float32).reshape(HIDDEN_SHAPE)

        # 4) run model
        X = [
            torch.from_numpy(depth).to(dev),
            torch.tensor([[des_vel]], dtype=torch.float32, device=dev),
            torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32, device=dev),
            (torch.from_numpy(h).to(dev), torch.from_numpy(c).to(dev))
        ]
        with torch.no_grad():
            vel, (h1, c1) = model(X)
            out_vel = vel.cpu().numpy().astype(np.float32).flatten()
            # pack reply: 3×float32 + h1,c1 as float16
            reply = bytearray()
            reply += out_vel.tobytes()
            reply += h1.cpu().numpy().astype(np.float16).tobytes()
            reply += c1.cpu().numpy().astype(np.float16).tobytes()

        sock.sendto(reply, addr)

if __name__ == '__main__':
    main()