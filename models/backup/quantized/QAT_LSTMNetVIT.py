import torch
import torch.nn as nn
import torch.nn.quantized as nnq


from models.quantized.QAT_ViT_Layers import QuantReadyTransformerEncoderLayer
from third_party.vitfly.models.model import refine_inputs
from models.quantized.ITA_utils import requantize_tensor, ita_partial_max

class MultiheadITAWithRequant(nn.Module):
    # This module now correctly simulates the ITA's integer-based pipeline
    def __init__(self, embed_dim, num_heads, params=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0
        assert params is not None
        self.params = params

        # These will be replaced by quantized layers during QAT
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_input, kv_input):
        B_q, N_q, _ = q_input.shape
        B_kv, N_kv, _ = kv_input.shape

        # Projections are done in float during QAT, then requantized
        Q_proj = self.q_proj(q_input)
        K_proj = self.k_proj(kv_input)
        V_proj = self.v_proj(kv_input)
        
        # Requantize to simulate hardware step
        Qp = requantize_tensor(Q_proj, self.params["mq"], self.params["sq"], self.params["zp"])
        Kp = requantize_tensor(K_proj, self.params["mk"], self.params["sk"], self.params["zp"])
        Vp = requantize_tensor(V_proj, self.params["mv"], self.params["sv"], self.params["zp"])

        Qp = Qp.reshape(B_q, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Kp = Kp.reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Vp = Vp.reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(Qp.float(), Kp.float().transpose(-2, -1))
        attn_logits_requant = requantize_tensor(attn_logits, self.params["ma"], self.params["sa"], self.params["zp"])
        attn_weights = ita_partial_max(attn_logits_requant.float(), k=8)
        context = torch.matmul(attn_weights, Vp.float())
        context_requant = requantize_tensor(context, self.params["mav"], self.params["sav"], self.params["zp"])
        context_requant = context_requant.permute(0, 2, 1, 3).reshape(B_q, N_q, self.embed_dim)
        
        final_proj = self.out_proj(context_requant.float())
        final = requantize_tensor(final_proj, self.params["mo"], self.params["so"], self.params["zp"])
        
        return final.float()

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