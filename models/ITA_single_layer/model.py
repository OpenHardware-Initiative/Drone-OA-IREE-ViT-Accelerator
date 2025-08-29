# ita_model_qat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from torch.nn.utils import spectral_norm

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# Assume ita_layers_qat.py is in the same directory or accessible
from models.ITA.layers import ITASelfAttention, ITAFeedForward, OverlapPatchMerging

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

class ITALSTMNetVIT_single_layer(nn.Module):
    """
    Model architecture with a single encoder layer and reduced embedding dimension.
    """
    def __init__(self):
        super().__init__()
        
        # --- MODIFIED: Reduced embedding dimension and single encoder layer ---
        self.E, self.S, self.P, self.F, self.H = 64, 128, 192, 256, 1
        
        # --- 1. CPU Pre-processing (Float) ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- MODIFIED: A single encoder block instead of a list ---
        # --- 2. ITA Accelerated Part (To be Quantized) ---
        self.attention_block = ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
        self.ffn_block = ITAFeedForward(embed_dim=self.E, ffn_dim=self.F)
        
        # --- CPU-bound Layers (Float) ---
        self.norm1 = nn.LayerNorm(self.E)
        self.norm2 = nn.LayerNorm(self.E)
        
        # --- 3. CPU Post-processing (Float) ---
        # --- MODIFIED: Recalculated decoder input size ---
        # New input size = self.E * self.S = 64 * 128 = 8192
        self.decoder = spectral_norm(nn.Linear(self.E * self.S, 512))
        
        # LSTM input size is 512 (from decoder) + 1 (desvel) + 4 (quat) = 517. No change needed.
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        
    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x, H, W = self.tokenizer(img_data)
        
        # --- MODIFIED: Replaced for-loop with direct calls for a single encoder layer ---
        # Attention sub-block
        attn_out = self.attention_block(x)
        x = x + attn_out # Residual connection
        x = self.norm1(x)

        # FFN sub-block
        ffn_out = self.ffn_block(x)
        x = x + ffn_out # Residual connection
        x = self.norm2(x)
        
        # --- Post-processing ---
        x = x.flatten(1)
        out = self.decoder(x)
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h