"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the models that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *

def refine_inputs(X):

    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        # X[2] = torch.Tensor([1, 0, 0, 0]).float()
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X



class LSTMNetVIT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

class ITALSTM(nn.Module):
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
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)

    def _encode(self, x):
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return embeds[1:]

    def _decode(self, encoded_features):
        out = torch.cat([self.pxShuffle(encoded_features[1]), self.up_sample(encoded_features[0])], dim=1)
        out = self.down_sample(out)
        return self.decoder(out.flatten(1))

    def forward(self, X):
        X = refine_inputs(X)
        x = X[0]

        encoded_features = self._encode(x)
        out = self._decode(encoded_features)
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
    
class ViT(nn.Module):
    """
    ViT+FC Network 
    Num Params: 3,101,199   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])        
        self.decoder = nn.Linear(4608, 512)
        self.nn_fc1 = spectral_norm(nn.Linear(517, 256))
        self.nn_fc2 = spectral_norm(nn.Linear(256, 3))
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        out = F.leaky_relu(self.nn_fc1(out))
        out = self.nn_fc2(out)

        return out, None

class ITAConformer(nn.Module):   
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
        self.decoder = nn.Linear(4608, 512)
        self.nn_fc1 = spectral_norm(nn.Linear(517, 256))
        self.nn_fc2 = spectral_norm(nn.Linear(256, 3))
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def _encode(self, x):
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return embeds[1:]

    def _decode(self, encoded_features):
        out = torch.cat([self.pxShuffle(encoded_features[1]), self.up_sample(encoded_features[0])], dim=1)
        out = self.down_sample(out)
        return self.decoder(out.flatten(1))

    def forward(self, X):
        X = refine_inputs(X)

        x = X[0]
        encoded_features = self._encode(x)
        out = self._decode(encoded_features)
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        out = F.leaky_relu(self.nn_fc1(out))
        out = self.nn_fc2(out)

        return out, None



if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")

    model = LSTMNetVIT().float()
    print("VITLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model2 = ITALSTM().float()
    print("ITAConformerLSTM: ")
    print(sum(p.numel() for p in model2.parameters() if p.requires_grad))
    model3 = ViT().float()
    print("ViT: ")
    print(sum(p.numel() for p in model3.parameters() if p.requires_grad))
    model4 = ITAConformer().float()
    print("ITAConformer: ")
    print(sum(p.numel() for p in model4.parameters() if p.requires_grad))
