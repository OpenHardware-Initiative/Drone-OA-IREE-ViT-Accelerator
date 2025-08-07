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
from models.ITA.ITA_layers import OverlapPatchMerging, ITASelfAttention, ITAFeedForward

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
    def __init__(self):
        super().__init__()
        
        # --- ITA Hardware Fixed Parameters ---
        self.E, self.S, self.P, self.F, self.H = 128, 128, 192, 256, 1
        
        # --- 1. CPU Pre-processing: Tokenizer ---
        self.tokenizer = OverlapPatchMerging(
                            in_channels=1, 
                            out_channels=self.E, 
                            patch_size=7, 
                            stride=2, 
                            padding=3, 
                            output_size=(8, 16)
                        )


        # --- 2. ITA Accelerated Part ---
        self.attention_blocks = nn.ModuleList([
            ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward(embed_dim=self.E, ffn_dim=self.F)
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
        
        self.quant_decoder = torch.quantization.QuantStub()
        self.dequant_out = torch.quantization.DeQuantStub()
        
        self.cat = nnq.FloatFunctional()
        self.add = nnq.FloatFunctional()
        
    def fuse_model(self):
        """
        Fuses operations for better QAT performance.
        Currently, there are no standard fusible layers in this architecture,
        but the method must exist to be called by the training script.
        """
        pass

    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x_float, H, W = self.tokenizer(img_data)
        
        for i in range(len(self.attention_blocks)):
            # Attention sub-block
            res_attn = x_float
            x_quant_attn = self.quant_attention[i](x_float)
            x_quant_attn = self.attention_blocks[i](x_quant_attn, H, W)
            x_float = self.dequant_attention[i](x_quant_attn)
            
            x_float = self.add.add(res_attn, x_float)
            x_float = self.norm1_layers[i](x_float)

            # FFN sub-block
            res_ffn = x_float
            x_quant_ffn = self.quant_ffn[i](x_float)
            x_quant_ffn = self.ffn_blocks[i](x_quant_ffn, H, W)
            x_float = self.dequant_ffn[i](x_quant_ffn)
            
            x_float = self.add.add(res_ffn, x_float)
            x_float = self.norm2_layers[i](x_float)
        
        x = x_float.flatten(1)
        out = self.decoder(x)
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h
    
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