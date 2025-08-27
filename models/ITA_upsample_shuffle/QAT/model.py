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
        
        # --- Feature FusAion Layers (Float) from LSTMNetVIT ---
        # NOTE: The target model's fusion depends on multi-stage inputs of different resolutions.
        # We will adapt this by using intermediate outputs from our single encoder stream.
        self.up_sample = nn.Upsample(size=(32, 48), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        
        # Input channels = (E/4 for pxShuffle) + (E for up_sample) = 128/4 + 128 = 32 + 128 = 160
        # Output channels = 12, to get a flattened size of 12 * 32 * 48 = 18432
        # This differs from the target's 4608 due to the single-stream encoder's feature map sizes.
        # Let's adjust the down_sample layer to produce a similar feature vector size.
        # We can use an adaptive pool to fix the output size before the decoder.
        self.down_sample = nn.Conv2d(160, 48, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 12)) # Output size -> 48 * 8 * 12 = 4608

        # --- 3. CPU Post-processing (Float) ---
        self.decoder = spectral_norm(nn.Linear(self.E * self.S, 512)) # E*S = 16384
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        
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