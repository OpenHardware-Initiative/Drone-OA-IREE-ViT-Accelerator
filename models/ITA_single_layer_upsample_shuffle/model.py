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

class ITALSTMNetVIT(nn.Module):
    """
    Model architecture with a single encoder layer and a multi-branch feature fusion block.
    """
    def __init__(self):
        super().__init__()
        
        self.E, self.S, self.P, self.F, self.H = 64, 128, 192, 256, 1
        
        # --- 1. CPU Pre-processing (Float) ---
        # Tokenizer outputting a feature map of effective size (8, 16)
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. ITA Accelerated Part (To be Quantized) ---
        self.attention_block = ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
        self.ffn_block = ITAFeedForward(embed_dim=self.E, ffn_dim=self.F)
        
        # --- CPU-bound Layers (Float) ---
        self.norm1 = nn.LayerNorm(self.E)
        self.norm2 = nn.LayerNorm(self.E)
        
        # --- 3. CPU Post-processing (Float) ---
        
        # --- Feature Fusion Layers (Float) ---
        # See mathematical derivation for these specific values.
        # Branch A: PixelShuffle upscales by 2x, reducing channels by 4x.
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        # Branch B: Upsample to match Branch A's spatial dimensions (16, 32).
        self.up_sample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        # Downsample conv layer to reduce concatenated channels (16+64=80) to 9.
        # This yields the target flattened size of 9 * 16 * 32 = 4608.
        self.down_sample = nn.Conv2d(in_channels=80, out_channels=9, kernel_size=3, padding=1)

        # --- Decoder and LSTM ---
        # Decoder input is now the target size 4608.
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        
        # LSTM input size remains 512 + 1 (desvel) + 4 (quat) = 517.
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        
    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        # --- 1. Tokenization ---
        # x shape: (B, 128, 64), H=8, W=16
        x, H, W = self.tokenizer(img_data)
        
        # --- 2. Single Transformer Encoder Block ---
        # Attention sub-block
        attn_out = self.attention_block(x)
        x = x + attn_out # Residual connection
        x = self.norm1(x)

        # FFN sub-block
        ffn_out = self.ffn_block(x)
        x = x + ffn_out # Residual connection
        x = self.norm2(x)
        
        # --- 3. Feature Fusion Block ---
        # Reshape token sequence to 2D feature map for conv/upsample operations
        B, S, E = x.shape
        # x_2d shape: (B, 64, 8, 16)
        x_2d = x.transpose(1, 2).view(B, E, H, W)
        
        # Branch A (PixelShuffle): (B, 64, 8, 16) -> (B, 16, 16, 32)
        x_shuffled = self.pxShuffle(x_2d)
        
        # Branch B (Upsample): (B, 64, 8, 16) -> (B, 64, 16, 32)
        x_upsampled = self.up_sample(x_2d)
        
        # Concatenate along channel dimension: (B, 16+64, 16, 32) -> (B, 80, 16, 32)
        x_fused = torch.cat([x_shuffled, x_upsampled], dim=1)
        
        # Downsample to target channel size: (B, 80, 16, 32) -> (B, 9, 16, 32)
        x_down = self.down_sample(x_fused)

        # --- 4. Decoder and LSTM ---
        # Flatten for the linear decoder: (B, 9*16*32) -> (B, 4608)
        x = x_down.flatten(1)
        
        out = self.decoder(x)
        # unsqueeze(0) for LSTM sequence format
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        # squeeze(0) to remove sequence dimension for final FC layer
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h