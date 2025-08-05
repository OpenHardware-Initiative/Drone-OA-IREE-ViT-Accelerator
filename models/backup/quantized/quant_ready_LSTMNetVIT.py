import torch
import torch.nn as nn
import torch.nn.quantized as nnq


from models.quantized.quant_ready_ViT_Layers import QuantReadyTransformerEncoderLayer
from third_party.vitfly.models.model import refine_inputs

class QuantReadyLSTMNetViT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            QuantReadyTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            QuantReadyTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = nn.Linear(4608, 512)
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = nn.Linear(128, 3)

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.concat = nnq.FloatFunctional()

    def fuse_model(self):
        pass

    def forward(self, X):

        X = refine_inputs(X)

        x = self.quant(X[0])    # quantize the input for the ViT
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = self.concat.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))

        x1_scaled = X[1]/10
        out = self.dequant(self.concat.cat([out, x1_scaled, X[2]], dim=1).float())    # dequantize the input to the LSTM 
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        
        out = self.quant(out)  # quantize the LSTM output
        out = self.nn_fc2(out)
        return self.dequant(out), h