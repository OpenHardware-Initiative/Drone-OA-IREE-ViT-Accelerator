# ita_model_qat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from torch.nn.utils import spectral_norm # Kept for reference, but we will remove its usage

# Required for QAT stubs
from torch.quantization import QuantStub, DeQuantStub

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the QAT-enabled versions of your custom layers
from models.ITA.QAT.layers import ITASelfAttention_QAT, ITAFeedForward_QAT, OverlapPatchMerging

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
    QAT-enabled model with a VARIABLE number of transformer layers.
    """
    def __init__(self, num_layers: int = 1): # Add num_layers argument
        super().__init__()
        
        self.num_layers = num_layers
        self.E, self.S, self.P, self.F, self.H = 64, 128, 192, 256, 1
        
        # --- 1. Float Pre-processing ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. Transformer Blocks (now in ModuleLists) ---
        # Create ModuleList containers to hold all the layers
        self.attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        # QAT specific modules
        self.quants1 = nn.ModuleList()
        self.dequants1 = nn.ModuleList()
        self.quants2 = nn.ModuleList()
        self.dequants2 = nn.ModuleList()
        self.add = nnq.FloatFunctional()

        # Populate the lists with the desired number of layers
        for _ in range(num_layers):
            self.attention_blocks.append(
                ITASelfAttention_QAT(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            )
            self.ffn_blocks.append(
                ITAFeedForward_QAT(embed_dim=self.E, ffn_dim=self.F)
            )
            self.norms1.append(nn.LayerNorm(self.E))
            self.norms2.append(nn.LayerNorm(self.E))
            
            # Add a pair of stubs for each layer to handle float LayerNorms
            self.quants1.append(QuantStub())
            self.dequants1.append(DeQuantStub())
            self.quants2.append(QuantStub())
            self.dequants2.append(DeQuantStub())

        # --- 3. Float Post-processing (No changes here) ---
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        # ... rest of the fusion, decoder, and LSTM layers ...
        self.decoder = nn.Linear(4608, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        
        # Branch B: Upsample to match Branch A's spatial dimensions
        self.up_sample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        # Downsample conv to reduce concatenated channels
        self.down_sample = nn.Conv2d(in_channels=(self.E // 4) + self.E, out_channels=9, kernel_size=3, padding=1)


    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x, H, W = self.tokenizer(img_data)
        
        # --- Loop through the transformer layers ---
        for i in range(self.num_layers):
            # Attention sub-block
            x_quant = self.quants1[i](x)
            attn_out_quant = self.attention_blocks[i](x_quant)
            x_quant_res = self.add.add(x_quant, attn_out_quant)
            x = self.dequants1[i](x_quant_res)
            x = self.norms1[i](x)

            # FFN sub-block
            x_quant = self.quants2[i](x)
            ffn_out_quant = self.ffn_blocks[i](x_quant)
            x_quant_res = self.add.add(x_quant, ffn_out_quant)
            x = self.dequants2[i](x_quant_res)
            x = self.norms2[i](x)
        
        # --- 3. Feature Fusion (Float) ---
        B, S, E = x.shape
        x_2d = x.transpose(1, 2).view(B, E, H, W)
        x_shuffled = self.pxShuffle(x_2d)
        x_upsampled = self.up_sample(x_2d)
        x_fused = torch.cat([x_shuffled, x_upsampled], dim=1)
        x_down = self.down_sample(x_fused)

        # --- 4. Decoder and LSTM (Float) ---
        x = x_down.flatten(1)
        out = self.decoder(x)
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h