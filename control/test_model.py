import torch
from model import *

def generate_dummy_input(traj_len=10):
    # Simulate a sequence of depth images (T, 1, 60, 90)
    depth_images = torch.randn(traj_len, 1, 60, 90)

    # Simulate desired velocities (T, 1)
    control_input = torch.rand(traj_len, 1)

    # Simulate quaternion orientations (T, 4)
    orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * traj_len).float()

    return [depth_images, control_input, orientation]

if __name__ == '__main__':
    vit_model = ViT().float()
    lstm_vit_model = LSTMNetVIT().float()

    ita_conformer = ITAConformer(efficient_attn=True).float()
    lstm_itaconformer_model = ITALSTM(efficient_attn=True).float()

    X = generate_dummy_input(traj_len=5)

    out_vit, _ = vit_model(X)
    print("Output shape (ViT):", out_vit.shape)

    out_vitlstm, _ = lstm_vit_model(X)
    print("Output shape (LSTMNetVIT):", out_vitlstm.shape)

    out_itaconformer, _ = ita_conformer(X)
    print("Output shape (ITAConformer):", out_itaconformer.shape)

    out_lstmitaconformer, _ = lstm_itaconformer_model(X)
    print("Output shape (LSTMNetITAConformer):", out_lstmitaconformer.shape)