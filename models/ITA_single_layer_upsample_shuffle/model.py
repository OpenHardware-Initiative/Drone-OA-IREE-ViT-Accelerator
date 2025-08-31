# ita_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the standard (non-QAT) versions of the custom layers
from models.ITA.layers import ITASelfAttention, ITAFeedForward, OverlapPatchMerging

def refine_inputs(X):
    """
    Ensures input tensors have the correct shape and format.
    Specifically, it handles missing quaternion data and resizes the input image.
    """
    # Ensure quaternion data exists
    if len(X) < 3 or X[2] is None:
        quat = torch.zeros((X[0].shape[0], 4), dtype=torch.float32, device=X[0].device)
        quat[:, 0] = 1 # Initialize with a valid unit quaternion (no rotation)
        if len(X) < 3: X.append(quat)
        else: X[2] = quat
    
    # Ensure image data has the correct spatial dimensions
    if X[0].shape[-2:] != (60, 90):
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear', align_corners=False)
    return X

class ITALSTMNetVIT(nn.Module):
    """
    A Vision Transformer (ViT) based model with a VARIABLE number of transformer layers
    and a multi-branch feature fusion block.
    """
    def __init__(self, num_layers: int = 1):
        super().__init__()
        
        self.num_layers = num_layers
        self.E, self.S, self.P, self.F, self.H = 64, 128, 192, 256, 1
        
        # --- 1. Pre-processing ---
        # Tokenizer converts the input image into a sequence of embedding vectors
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. Transformer Blocks (Variable number of layers) ---
        # Use nn.ModuleList to hold the layers, allowing for a flexible depth
        self.attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        # Populate the lists with the desired number of transformer encoder layers
        for _ in range(self.num_layers):
            self.attention_blocks.append(
                ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            )
            self.ffn_blocks.append(
                ITAFeedForward(embed_dim=self.E, ffn_dim=self.F)
            )
            self.norms1.append(nn.LayerNorm(self.E))
            self.norms2.append(nn.LayerNorm(self.E))
            
        # --- 3. Feature Fusion Block ---
        # Branch A: PixelShuffle upscales by 2x, reducing channels by 4x. (64/4=16)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        # Branch B: Upsample to match Branch A's spatial dimensions (16, 32).
        self.up_sample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        # Downsample conv to reduce concatenated channels (16+64=80) to 9.
        # This yields the target flattened size of 9 * 16 * 32 = 4608.
        self.down_sample = nn.Conv2d(in_channels=(self.E // 4) + self.E, out_channels=9, kernel_size=3, padding=1)

        # --- 4. Decoder and LSTM ---
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        # LSTM input size is 512 (from decoder) + 1 (additional_data) + 4 (quat_data) = 517
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        
    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        # --- 1. Tokenization ---
        # x shape: (B, 128, 64), H=8, W=16
        x, H, W = self.tokenizer(img_data)
        
        # --- 2. Transformer Encoder Blocks ---
        # Sequentially pass the data through each transformer layer
        for i in range(self.num_layers):
            # Attention sub-block with residual connection
            attn_out = self.attention_blocks[i](x)
            x = x + attn_out
            x = self.norms1[i](x)

            # Feed-Forward Network (FFN) sub-block with residual connection
            ffn_out = self.ffn_blocks[i](x)
            x = x + ffn_out
            x = self.norms2[i](x)
        
        # --- 3. Feature Fusion Block ---
        # Reshape token sequence back to a 2D feature map for convolutional operations
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
        # Concatenate with additional sensor data and add a sequence dimension for the LSTM
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        # Manage LSTM hidden state
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        
        # Remove sequence dimension and pass through final fully connected layer
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h