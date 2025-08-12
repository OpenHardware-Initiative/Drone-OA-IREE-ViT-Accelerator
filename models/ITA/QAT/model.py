# ita_model_qat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq

# Assume ita_layers_qat.py is in the same directory or accessible
from .layers import ITASelfAttention_QAT, ITAFeedForward_QAT, OverlapPatchMerging

def refine_inputs(X):
    # (Same as your original function)
    if len(X) < 3 or X[2] is None:
        quat = torch.zeros((X[0].shape[0], 4), dtype=torch.float32, device=X[0].device)
        quat[:, 0] = 1
        if len(X) < 3: X.append(quat)
        else: X[2] = quat
    if X[0].shape[-2:] != (60, 90):
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear', align_corners=False)
    return X

class ITALSTMNetVIT_QAT(nn.Module):
    """
    Model architecture for mixed-precision QAT.
    Attention and FFN blocks are quantized, the rest remains float.
    """
    def __init__(self):
        super().__init__()
        
        self.E, self.S, self.P, self.F, self.H = 128, 128, 192, 256, 1
        
        # --- 1. CPU Pre-processing (Float) ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. ITA Accelerated Part (To be Quantized) ---
        self.attention_blocks = nn.ModuleList([
            ITASelfAttention_QAT(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward_QAT(embed_dim=self.E, ffn_dim=self.F)
            for _ in range(2)
        ])
        
        # --- CPU-bound Layers (Float) ---
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        # --- 3. CPU Post-processing (Float) ---
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        
        # Functional wrappers for operations between float and quant regions
        self.add = nnq.FloatFunctional()
        self.cat = nnq.FloatFunctional()
        
    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x, H, W = self.tokenizer(img_data)
        
        for i in range(len(self.attention_blocks)):
            # Attention sub-block
            # The input 'x' is float. QAT will insert quantize ops automatically.
            attn_out = self.attention_blocks[i](x)
            # The output 'attn_out' is dequantized back to float by QAT.
            x = self.add.add(x, attn_out) # Residual connection
            x = self.norm1_layers[i](x)

            # FFN sub-block
            ffn_out = self.ffn_blocks[i](x)
            x = self.add.add(x, ffn_out) # Residual connection
            x = self.norm2_layers[i](x)
        
        x = x.flatten(1)
        out = self.decoder(x)
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h