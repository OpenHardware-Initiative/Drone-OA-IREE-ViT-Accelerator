import torch
import sys
import os
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.testing.ITA_Layers import OverlapPatchMerging, ITASelfAttention, ITAFeedForward

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
    """Final model architecture ready for QAT and hardware-compliant simulation."""
    def __init__(self, params, qat_mode=False):
        super().__init__()
        self.qat_mode = qat_mode
        
        # --- ITA Hardware Fixed Parameters ---
        self.E, self.S, self.P, self.F, self.H = 128, 64, 192, 256, 1
        
        # --- 1. CPU Pre-processing: Tokenizer (SegFormer-style) ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, 
            out_channels=self.E,
            patch_size=7, 
            stride=2,
            padding=3,
            output_size=(8, 8)  # Guarantees H=8, W=8, so S = 64
        )

        # --- 2. ITA Accelerated Part (Modules defined separately) ---
        self.attention_blocks = nn.ModuleList([
            ITASelfAttention(embed_dim=self.E, proj_dim=self.P, num_heads=self.H, params=params)
            for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward(embed_dim=self.E, ffn_dim=self.F, params=params, qat_mode=self.qat_mode)
            for _ in range(2)
        ])
        
        # --- CPU-bound Normalization Layers ---
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        # --- 3. CPU Post-processing: Decoder and LSTM Head ---
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        
        # --- Quantization Utilities ---
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.cat = nnq.FloatFunctional()
        self.add = nnq.FloatFunctional()

    def fuse_model(self):
        """Fuses operations for better QAT performance."""
        pass
    
    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        # --- Part 1: Run on CPU ---
        x_float, H, W = self.tokenizer(img_data)
        
        # --- Part 2: Loop with correct CPU/ITA partitioning ---
        for i in range(len(self.attention_blocks)):
            # --- Attention Sub-layer (ITA) ---
            attn_out_float = self.dequant(self.attention_blocks[i](self.quant(x_float), H, W))
            
            # --- Add & Norm 1 (CPU) ---
            res1_float = self.add.add(x_float, attn_out_float)
            norm1_out_float = self.norm1_layers[i](res1_float)

            # --- FFN Sub-layer (ITA) ---
            ffn_out_float = self.dequant(self.ffn_blocks[i](self.quant(norm1_out_float), H, W))

            # --- Add & Norm 2 (CPU) ---
            res2_float = self.add.add(norm1_out_float, ffn_out_float)
            x_float = self.norm2_layers[i](res2_float)
        
        # --- Part 3: Run on CPU ---
        x = x_float.flatten(1)
        out = self.decoder(x)
        
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1)
        out_cat = out_cat.unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
            
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h