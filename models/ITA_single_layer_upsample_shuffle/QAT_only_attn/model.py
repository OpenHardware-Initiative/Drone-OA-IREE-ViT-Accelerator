# ita_model_qat.py

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

# --- MODIFIED IMPORTS ---
# We now import the QAT attention layer but the standard float feed-forward layer
from models.ITA.QAT.layers import ITASelfAttention_QAT, OverlapPatchMerging
from models.ITA.layers import ITAFeedForward # Using the standard float version

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
    QAT-enabled model where ONLY the attention block is quantized.
    The Feed-Forward Network (FFN) remains a standard float module.
    """
    def __init__(self, num_layers: int = 1):
        super().__init__()
        
        self.num_layers = num_layers
        self.E, self.S, self.P, self.F, self.H = 64, 128, 192, 256, 1
        
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        self.attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        # --- MODIFICATION: Only one set of stubs is needed for the attention block ---
        self.quants1 = nn.ModuleList([QuantStub() for _ in range(num_layers)])
        self.dequants1 = nn.ModuleList([DeQuantStub() for _ in range(num_layers)])
        self.add = nnq.FloatFunctional() # Kept for the quantized attention addition

        for _ in range(num_layers):
            # Attention block is the QAT version
            self.attention_blocks.append(
                ITASelfAttention_QAT(embed_dim=self.E, proj_dim=self.P, num_heads=self.H)
            )
            # --- MODIFICATION: FFN block is the standard float version ---
            self.ffn_blocks.append(
                ITAFeedForward(embed_dim=self.E, ffn_dim=self.F)
            )
            self.norms1.append(nn.LayerNorm(self.E))
            self.norms2.append(nn.LayerNorm(self.E))

        # --- Post-processing layers remain the same ---
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.decoder = nn.Linear(4608, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)
        self.up_sample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        self.down_sample = nn.Conv2d(in_channels=(self.E // 4) + self.E, out_channels=9, kernel_size=3, padding=1)

    def forward(self, X):
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]
        x, H, W = self.tokenizer(img_data)
        
        for i in range(self.num_layers):
            # --- Attention sub-block (Quantized) ---
            x_quant = self.quants1[i](x)
            attn_out_quant = self.attention_blocks[i](x_quant)
            x_quant_res = self.add.add(x_quant, attn_out_quant)
            x = self.dequants1[i](x_quant_res)
            x = self.norms1[i](x)

            # --- FFN sub-block (Standard Float Operation) ---
            # No Quant/DeQuant stubs are used here.
            ffn_out = self.ffn_blocks[i](x)
            x = x + ffn_out # Standard float residual connection
            x = self.norms2[i](x)
        
        # --- Feature Fusion, Decoder, and LSTM remain the same ---
        B, S, E = x.shape
        x_2d = x.transpose(1, 2).view(B, E, H, W)
        x_shuffled = self.pxShuffle(x_2d)
        x_upsampled = self.up_sample(x_2d)
        x_fused = torch.cat([x_shuffled, x_upsampled], dim=1)
        x_down = self.down_sample(x_fused)

        x = x_down.flatten(1)
        out = self.decoder(x)
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h