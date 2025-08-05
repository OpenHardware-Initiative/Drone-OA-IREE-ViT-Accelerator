# ita_model_export.py

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

# Import export-specific layers and the standard tokenizer
from models.testing.ITA_layers import OverlapPatchMerging
from models.testing.export.ITA_layers_export import ITASelfAttention_Export, ITAFeedForward_Export


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

class ITALSTMNetVIT_Export(nn.Module):
    """Model architecture configured for EXPORTING intermediate tensors."""
    def __init__(self):
        super().__init__()
        
        self.E, self.S, self.P, self.F, self.H = 128, 64, 192, 256, 1
        
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 8)
        )

        self.attention_blocks = nn.ModuleList([
            ITASelfAttention_Export(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward_Export(embed_dim=self.E, ffn_dim=self.F)
            for _ in range(2)
        ])
        
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        
        self.quant_attention = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        self.dequant_attention = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])
        self.quant_ffn = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        self.dequant_ffn = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])
        
        self.cat = nnq.FloatFunctional()
        self.add = nnq.FloatFunctional()

    def forward_with_intermediates(self, X):
        """
        A special forward pass that returns the ground truth intermediate 
        tensors from the converted PyTorch model for verification.
        """
        X = refine_inputs(X)
        img_data, _, _ = X[0], X[1], X[2]
        
        intermediates_list = []
        x_float, H, W = self.tokenizer(img_data)
        
        for i in range(len(self.attention_blocks)):
            attn_block = self.attention_blocks[i]
            ffn_block = self.ffn_blocks[i]
            
            # --- Attention Block ---
            q_in_attn = self.quant_attention[i](x_float)
            attn_out, attn_intermediates = attn_block(q_in_attn, H, W)
            attn_intermediates['Q_in'] = q_in_attn.int_repr()
            
            attn_out_float = self.dequant_attention[i](attn_out)
            res1_float = self.add.add(x_float, attn_out_float)
            norm1_out_float = self.norm1_layers[i](res1_float)
            
            # --- FFN Block ---
            q_in_ffn = self.quant_ffn[i](norm1_out_float)
            ffn_out, ffn_intermediates = ffn_block(q_in_ffn, H, W)
            ffn_intermediates['FF_in'] = q_in_ffn.int_repr()

            block_intermediates = {**attn_intermediates, **ffn_intermediates}
            intermediates_list.append(block_intermediates)
            
            # --- Prepare for Next Iteration ---
            ffn_out_float = self.dequant_ffn[i](ffn_out)
            res2_float = self.add.add(norm1_out_float, ffn_out_float)
            x_float = self.norm2_layers[i](res2_float)
                
        return intermediates_list