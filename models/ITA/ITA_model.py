# ita_model.py

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

# Import standard layers
from models.testing.ITA_layers import OverlapPatchMerging, ITASelfAttention, ITAFeedForward

def refine_inputs(X):
    """Pre-processes the input data to the required shape."""
    if len(X) < 3 or X[2] is None:
        quat = torch.zeros((X[0].shape[0], 4), dtype=torch.float32, device=X[0].device)
        quat[:, 0] = 1
        if len(X) < 3: X.append(quat)
        else: X[2] = quat
    if X[0].shape[-2:] != (60, 90):
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear', align_corners=False)
    return X

class ITALSTMNetVIT(nn.Module):
    """Final model architecture for QAT and standard inference."""
    def __init__(self, params):
        super().__init__()
        
        # --- ITA Hardware Fixed Parameters ---
        self.E, self.S, self.P, self.F, self.H = 128, 64, 192, 256, 1
        
        # --- 1. CPU Pre-processing: Tokenizer ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 8)
        )

        # --- 2. ITA Accelerated Part ---
        self.attention_blocks = nn.ModuleList([
            ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H, params=params)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward(embed_dim=self.E, ffn_dim=self.F, params=params)
            for _ in range(2)
        ])
        
        # --- CPU-bound Normalization Layers ---
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        # --- 3. CPU Post-processing: Decoder and LSTM Head ---
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        
        # --- Quantization Stubs and Functional Wrappers ---
        self.quant_attention = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        self.dequant_attention = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])
        self.quant_ffn = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        self.dequant_ffn = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])
        
        self.cat = nnq.FloatFunctional()
        self.add = nnq.FloatFunctional()

    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        # Part 1: CPU Tokenizer
        x_float, H, W = self.tokenizer(img_data)
        
        # Part 2: Main Transformer Loop (CPU/ITA Partitioning)
        for i in range(len(self.attention_blocks)):
            # Attention Sub-layer (ITA)
            q_in_attn = self.quant_attention[i](x_float)
            attn_out = self.attention_blocks[i](q_in_attn, H, W)
            attn_out_float = self.dequant_attention[i](attn_out)
            
            # Add & Norm 1 (CPU)
            res1_float = self.add.add(x_float, attn_out_float)
            norm1_out_float = self.norm1_layers[i](res1_float)

            # FFN Sub-layer (ITA)
            q_in_ffn = self.quant_ffn[i](norm1_out_float)
            ffn_out = self.ffn_blocks[i](q_in_ffn, H, W)
            ffn_out_float = self.dequant_ffn[i](ffn_out)

            # Add & Norm 2 (CPU)
            res2_float = self.add.add(norm1_out_float, ffn_out_float)
            x_float = self.norm2_layers[i](res2_float)
        
        # Part 3: CPU Decoder
        x = x_float.flatten(1)
        out = self.decoder(x)
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h