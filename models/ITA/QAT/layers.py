import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QConfig, FusedMovingAvgObsFakeQuantize

from .ITA_softmax import IntegerApproximatedSoftmax


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



class OverlapPatchMerging(nn.Module):
    """
    Implementation of overlap patch merging. This is already well-suited
    for export, as it uses a fixed output size.
    """
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, output_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, 
            stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(out_channels)
        self.output_size = output_size

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W
    
class ITAFeedForward_QAT(nn.Module):
    """Feed-Forward Network prepared for QAT with internal stubs."""
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        # Layers that will be quantized
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.identity = nn.Identity()
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        # Explicit quantization boundaries
        self.quant = torch.ao.quantization.QuantStub(ita_symmetric_qconfig)
        self.dequant = torch.ao.quantization.DeQuantStub(ita_symmetric_qconfig)

    def forward(self, x):
        # 1. Quantize the float input
        x = self.quant(x)
        
        # 2. Perform operations in the quantized domain
        x = self.fc1(x)
        x = self.identity(x)
        x = self.activation(x)
        
        x = self.identity(x)
        x = self.fc2(x)
        
        # 3. Dequantize the output back to float
        x = self.dequant(x)
        return x

class ITASelfAttention_QAT(nn.Module):
    """Self-Attention block for QAT with internal stubs and uint8 softmax."""
    def __init__(self, embed_dim, proj_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        
        # Layers that will be quantized
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        
        #self.softmax = nn.Softmax(dim=-1)
        self.custom_softmax  = IntegerApproximatedSoftmax(dim=-1)


        self.matmul1 = FloatFunctional()
        self.matmul2 = FloatFunctional()

        # Explicit quantization boundaries
        self.quant = torch.ao.quantization.QuantStub(ita_symmetric_qconfig)
        self.dequant = torch.ao.quantization.DeQuantStub(ita_symmetric_qconfig)

    def forward(self, x):
        # 1. Quantize float input
        x_quant = self.quant(x)
        
        # 2. Operations in the quantized domain
        Q = self.q_proj(x_quant)
        K = self.k_proj(x_quant)
        V = self.v_proj(x_quant)

        B, N, _ = x.shape # Use original float input shape for reshaping
        Q_r = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = self.matmul1.matmul(Q_r, K_r.transpose(-2, -1))
        #attn_logits = self.dequant_logits(attn_logits)
        attn_weights = self.custom_softmax(attn_logits)
        #attn_weights = self.quant_softmax_out(attn_weights_float)
        
        context = self.matmul2.matmul(attn_weights, V_r)
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        
        output_quant = self.out_proj(context)
        
        # 3. Dequantize output back to float
        output_float = self.dequant(output_quant)
        return output_float