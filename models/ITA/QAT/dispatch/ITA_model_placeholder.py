import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

from models.ITA.ITA_layers import OverlapPatchMerging
from models.ITA.dispatch.ITA_layers_placeholder import ITAFeedForward_Placeholder, ITASelfAttention_Placeholder

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


class ITALSTMNetVIT_Placeholder_Export(nn.Module):
    """
    Final EXPORT model architecture that uses the placeholder layers.
    This version is designed to be exported to ONNX for IREE compilation.
    """
    def __init__(self):
        super().__init__()

        self.E, self.S, self.P, self.F, self.H = 128, 128, 192, 256, 1

        # These layers will run on the CPU and need their weights loaded.
        self.tokenizer = OverlapPatchMerging(
                            in_channels=1, 
                            out_channels=self.E, 
                            patch_size=7, 
                            stride=2, 
                            padding=3, 
                            output_size=(8, 16)
                        )
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)

        # --- Use the Placeholder versions of the blocks ---
        self.attention_blocks = nn.ModuleList([
            ITASelfAttention_Placeholder() for _ in range(2)
        ])
        self.ffn_blocks = nn.ModuleList([
            ITAFeedForward_Placeholder() for _ in range(2)
        ])

        # Quant/Dequant stubs to manage the boundary between CPU (float) and Accelerator (int8)
        self.quant_attention = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        # The dequant stubs are no longer needed here, as the custom op handles it.
        # self.dequant_attention = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])
        self.quant_ffn = nn.ModuleList([torch.quantization.QuantStub() for _ in range(2)])
        # self.dequant_ffn = nn.ModuleList([torch.quantization.DeQuantStub() for _ in range(2)])

        self.cat = nnq.FloatFunctional()
        self.add = nnq.FloatFunctional()

    def forward(self, image, additional_data, quat_data, hidden_in, cell_in):
        # By making inputs explicit, we can remove the non-traceable `refine_inputs` call.
        # The TracerWarning will disappear.
        img_data = image

        x_float, H, W = self.tokenizer(img_data)

        for i in range(len(self.attention_blocks)):
            # --- Attention Block Boundary ---
            res_attn = x_float
            x_quant_attn = self.quant_attention[i](x_float)
            attn_out_float = self.attention_blocks[i](x_quant_attn)
            x_float = self.add.add(res_attn, attn_out_float)
            x_float = self.norm1_layers[i](x_float)

            # --- FFN Block Boundary ---
            res_ffn = x_float
            x_quant_ffn = self.quant_ffn[i](x_float)
            ffn_out_float = self.ffn_blocks[i](x_quant_ffn)
            x_float = self.add.add(res_ffn, ffn_out_float)
            x_float = self.norm2_layers[i](x_float)

        x = x_float.flatten(1)
        out = self.decoder(x)
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)

        # Explicitly create the hidden state tuple for the LSTM
        hidden_state = (hidden_in, cell_in)
        
        # Capture both LSTM outputs and the new hidden/cell states
        out_lstm, (hidden_out, cell_out) = self.lstm(out_cat, hidden_state)
        
        # This logic handles the case where the sequence length is 1
        if out_lstm.shape[0] == 1:
             out_lstm = out_lstm.squeeze(0)

        out_final = self.nn_fc2(out_lstm)

        # Return all three outputs as defined in your export call
        return out_final, hidden_out, cell_out