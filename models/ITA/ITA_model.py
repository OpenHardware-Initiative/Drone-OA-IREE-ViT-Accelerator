# ita_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

# Import float-specific layers
from .ITA_layers import OverlapPatchMerging, ITASelfAttentionFloat, ITAFeedForwardFloat, ITASoftmax 

def refine_inputs(X):
    """
    Pre-processes the input data.
    BEST PRACTICE: This logic should ideally be in your DataLoader to ensure
    the model always receives a static-sized tensor, simplifying the graph for MLIR.
    """
    if len(X) < 3 or X[2] is None:
        quat = torch.zeros((X[0].shape[0], 4), dtype=torch.float32, device=X[0].device)
        quat[:, 0] = 1
        X = list(X)
        if len(X) < 3: X.append(quat)
        else: X[2] = quat
    if X[0].shape[-2:] != (60, 90):
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear', align_corners=False)
    return X

class ITALSTMNetVITFloat(nn.Module):
    """
    Final model architecture redesigned for float training and MLIR export.
    - REMOVED: All QuantStub, DeQuantStub, and nnq.FloatFunctional modules.
    - SIMPLIFIED: The forward pass is now a direct, sequential float computation.
    """
    def __init__(self):
        super().__init__()
        self.E, self.S, self.P, self.F, self.H = 128, 128, 192, 256, 1
        
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        self.attention_blocks = nn.ModuleList([
            ITASelfAttentionFloat(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForwardFloat(embed_dim=self.E, ffn_dim=self.F)
            for _ in range(2)
        ])
        
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)

    def forward(self, X):
        # NOTE: Keeping refine_inputs here for consistency, but moving it to the
        # dataloader is recommended for a truly static graph.
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x, H, W = self.tokenizer(img_data)
        
        for i in range(len(self.attention_blocks)):
            # Attention sub-block with residual connection
            res_attn = x
            x_attn = self.attention_blocks[i](x) # Simplified call
            x = self.norm1_layers[i](res_attn + x_attn)

            # FFN sub-block with residual connection
            res_ffn = x
            x_ffn = self.ffn_blocks[i](x) # Simplified call
            x = self.norm2_layers[i](res_ffn + x_ffn)
        
        x = x.flatten(1)
        out = self.decoder(x)
        
        # Use torch.cat directly
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h

