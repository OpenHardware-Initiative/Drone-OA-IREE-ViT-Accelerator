import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm

from models.quantized.quant_ready_ViT_Layers import QuantReadyITAEncoderLayer, QuantReadyMixFFN

from third_party.vitfly.models.ViTsubmodules import *
from third_party.vitfly.models.model import refine_inputs



class QuantReadyITALSTM(nn.Module):
    """
    ITAConformer+LSTM Network
    """
    def __init__(self, itaparameters=None, efficient_attn=True):
        super().__init__()
        
        if itaparameters == None:
            itaparameters = {
            "mq": 1.0, "sq": 0, # multiplier_q and shift_q
            "mk": 1.0, "sk": 0, # multiplier_k and shift_k
            "mv": 1.0, "sv": 0, # multiplier_v and shift_v
            "ma": 1.0, "sa": 0, # multiplier_attn and shift_attn
            "mav": 1.0, "sav": 0, # multiplier_av and shift_av
            "mo": 1.0, "so": 0, # multiplier_o and shift_o
            "mf": 1.0, "sf": 0, # multiplier_final and shift_final
            "zp": 0, # zero_point
            }

        self.encoder_blocks = nn.ModuleList([
            QuantReadyITAEncoderLayer(1, 32, patch_size=7, stride=4, padding=3,
                               n_layers=2, reduction_ratio=8, num_heads=1,
                               expansion_factor=8, embed_dim=32,
                               efficient_attn=efficient_attn, itaparameters=itaparameters),
            QuantReadyITAEncoderLayer(32, 64, patch_size=3, stride=2, padding=1,
                               n_layers=2, reduction_ratio=4, num_heads=2,
                               expansion_factor=8, embed_dim=64,
                               efficient_attn=efficient_attn, itaparameters=itaparameters)
        ])

        self.decoder = nn.Linear(4608, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128,
                            num_layers=3, dropout=0.1)
        self.nn_fc2 = nn.Linear(128, 3)

        self.up_sample = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.concat = nnq.FloatFunctional()

    def _encode(self, x):
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return embeds[1:]

    def _decode(self, encoded_features):
        out = torch.cat([self.pxShuffle(encoded_features[1]), self.up_sample(encoded_features[0])], dim=1)
        out = self.down_sample(out)
        return self.decoder(out.flatten(1))
    
    def fuse_model(self):
        pass

    def forward(self, X):
        X = refine_inputs(X)
        x = X[0]

        x = self.quant(x)

        encoded_features = self._encode(x)
        out = self._decode(encoded_features)
        # Each out[i]: (T, C, H, W)
        # Flatten per timestep: (T, 12, 16, 24) -> (T, 4608)
        # Concat additional inputs: (T, 512) + (1, T, 1) + (1, T, 4) => (T, 517)
        x1_scaled = X[1] / 10.0
        #out = self.dequant(self.concat.cat([out, x1_scaled, X[2]], dim=1).float())
        out = self.concat.cat([out, x1_scaled, X[2]], dim=1).float()
        out = out.unsqueeze(1)  # Ensure correct shape for LSTM in QAT mode
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
        out = out.squeeze(1)  # Restore original shape
        
        #out = self.quant(out)
        out = self.nn_fc2(out)
        return self.dequant(out), h