# ita_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torch.ao.nn.quantized as nnq
from torch.ao.quantization.qconfig import QConfig, FusedMovingAvgObsFakeQuantize
from torch.ao.quantization import QuantStub, DeQuantStub
from typing import Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

# Import standard layers
from torch_model.ITA_layers import OverlapPatchMerging, ITASelfAttention, ITAFeedForward



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
        
        ita_symmetric_qconfig = QConfig(
            activation=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            ),
            weight=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            )
        )
        
        self.quant_tokenizer = QuantStub(ita_symmetric_qconfig)
        
        # --- Quantization Stubs and Functional Wrappers ---
        self.quant_attention = nn.ModuleList([QuantStub(ita_symmetric_qconfig) for _ in range(2)])
        self.dequant_attention = nn.ModuleList([DeQuantStub(ita_symmetric_qconfig) for _ in range(2)])
        self.quant_ffn = nn.ModuleList([QuantStub(ita_symmetric_qconfig) for _ in range(2)])
        self.dequant_ffn = nn.ModuleList([DeQuantStub(ita_symmetric_qconfig) for _ in range(2)])
        
        #self.quant_decoder = QuantStub(ita_symmetric_qconfig)
        self.dequant_decoder = DeQuantStub(ita_symmetric_qconfig)
        
        self.cat = torch.cat
        self.add = torch.add
    
        
    def fuse_model(self):
        """
        Fuses operations for better QAT performance.
        Currently, there are no standard fusible layers in this architecture,
        but the method must exist to be called by the training script.
        """
        pass

    def forward(
        self, 
        img_tensor,
        additional_data,
        quat_tensor,
        hidden_state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        

        q_img_tensor = self.quant_tokenizer(img_tensor)
        
        q_x, H, W = self.tokenizer(q_img_tensor)
        
        for i in range(len(self.attention_blocks)):
            # Attention sub-block
            q_res_attn = q_x
            q_x_attn = self.attention_blocks[i](q_x, H, W)
            
            q_x = self.add(q_res_attn, q_x_attn)
            q_x = self.norm1_layers[i](q_x)

            # FFN sub-block
            q_res_ffn = q_x
            q_x_ffn = self.ffn_blocks[i](q_x, H, W)
            
            q_x = self.add(q_res_ffn, q_x_ffn)
            q_x = self.norm2_layers[i](q_x)
        
        x = q_x.flatten(1)
        q_out = self.decoder(x)
        out = self.dequant_decoder(q_out)
        out_cat = self.cat([out, additional_data / 10.0, quat_tensor], dim=1).unsqueeze(0)
        
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h
    
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
    
class ITALSTMNetVITWrapper(nn.Module):
    """
    A wrapper for the ITALSTMNetVIT model to maintain compatibility with
    training scripts that pass a single list of tensors.
    
    This module's forward method accepts a list `X` and unpacks it
    to call the inner model with the correct keyword arguments.
    """
    def __init__(self):
        super().__init__()
        # Instantiate the actual, ONNX-exportable model
        self.model = ITALSTMNetVIT()

    def forward(self, X: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpacks the input list X and calls the underlying model.

        Args:
            X (list): A list of tensors expected in the format:
                      [image_tensor, additional_data, quat_tensor, optional_hidden_state]
        
        Returns:
            The output of the underlying model.
        """
        
        X = refine_inputs(X)
        
        # Unpack the list from the dataloader/training script
        image_tensor = X[0]
        additional_data = X[1]
        quat_tensor = X[2] if len(X) > 2 else None
        hidden_state = X[3] if len(X) > 3 else None
        
        # Call the actual model with the correct, named arguments
        return self.model(
            img_tensor=image_tensor,
            additional_data=additional_data,
            quat_tensor=quat_tensor,
            hidden_state=hidden_state
        )