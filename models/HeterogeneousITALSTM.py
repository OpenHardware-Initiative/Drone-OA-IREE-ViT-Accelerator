import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm

from ITA_ViTSubmodules import MiXITAEncoderLayer

from third_party.vitfly.models.ViTsubmodules import *
from third_party.vitfly.models.model import refine_inputs

class HeterogeneousITALSTM(nn.Module):
    """
    ViT + CNN + LSTM
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
            MiXITAEncoderLayer(1, 32, patch_size=7, stride=4, padding=3,
                               n_layers=2, reduction_ratio=8, num_heads=1,
                               expansion_factor=8, embed_dim=32,
                               efficient_attn=efficient_attn, itaparameters=itaparameters),
            MiXITAEncoderLayer(32, 64, patch_size=3, stride=2, padding=1,
                               n_layers=2, reduction_ratio=4, num_heads=2,
                               expansion_factor=8, embed_dim=64,
                               efficient_attn=efficient_attn, itaparameters=itaparameters)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = nn.LSTM(input_size=517, hidden_size=128,
                            num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.fuse_conv = nn.Conv2d(60, 12, 3, padding=1)

        self.conv_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # (N,32,60,90)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # (N,32,30,45)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (N,64,30,45)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # (N,64,15,23)

            nn.Conv2d(64, 12, kernel_size=3, stride=1, padding=1),  # (N,12,15,23)
            nn.ReLU(),

            nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)  # (N,12,16,24)
        )


    def _encode(self, x):
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return embeds[1:]

    def _decode(self, encoded_features, raw_input):

        conv_features = self.conv_feature_extractor(raw_input)

        out = torch.cat([self.pxShuffle(encoded_features[1]), self.up_sample(encoded_features[0]), conv_features], dim=1)
        out = self.fuse_conv(out)
        return self.decoder(out.flatten(1))


    def forward(self, X):
        X = refine_inputs(X)
        x = X[0]

        encoded_features = self._encode(x)
        out = self._decode(encoded_features, x)
        # Each out[i]: (T, C, H, W)
        # Flatten per timestep: (T, 12, 16, 24) -> (T, 4608)
        # Concat additional inputs: (T, 512) + (1, T, 1) + (1, T, 4) => (T, 517)
        out = torch.cat([out, X[1].squeeze(0)/10, X[2].squeeze(0)], dim=1).float()
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h