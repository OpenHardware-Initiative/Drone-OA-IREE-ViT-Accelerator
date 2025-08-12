# ita_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization.qconfig import QConfig, FusedMovingAvgObsFakeQuantize
from torch.ao.quantization import QuantStub, DeQuantStub

# --------------------------------------
# Hardware-Compliant Helper Modules
# --------------------------------------

class MatMulTranspose(nn.Module):
    """
    A QAT-compatible module that performs matrix multiplication with the second
    argument transposed (B.T). This allows QAT observers to be attached.
    """
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b.transpose(-2, -1))
    
class MatMul(nn.Module):
    """
    A QAT-compatible module that performs standard matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


class ITASoftmax(nn.Module):
    """A hardware-compliant streaming Softmax implementation."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, n_heads, seq_length, seq_len_kv = x.shape
        x_3d = x.reshape(B * n_heads, seq_length, seq_len_kv).to(torch.int32)
        num_effective_heads = B * n_heads

        width, bits, range_scale = 16, 8, 32
        groups = seq_len_kv // width
        eps_max = range_scale * bits / (2**bits)

        exp_partial_sum = torch.zeros((num_effective_heads, seq_length), dtype=torch.int32, device=x.device)
        global_max = torch.full((num_effective_heads, seq_length), -128, dtype=torch.int8, device=x.device)

        for i in range(groups):
            chunk = x_3d[..., i * width:(i + 1) * width]
            current_max = torch.max(chunk, dim=-1)[0]
            
            update_mask = current_max > global_max
            max_shift = torch.floor((current_max - global_max) * eps_max + 0.5).to(torch.int32)
            
            shift_sum = torch.zeros_like(exp_partial_sum)
            shift_sum[update_mask] = max_shift[update_mask]
            global_max[update_mask] = current_max[update_mask].to(torch.int8)

            diff = global_max.unsqueeze(-1) - chunk
            shift = torch.floor(diff * eps_max + 0.5).to(torch.int32)
            exp_sum = torch.sum(2**bits >> shift, dim=-1).to(torch.int32)
            exp_partial_sum = (exp_partial_sum >> shift_sum) + exp_sum

        exp_partial_sum_inverse = torch.floor((2**bits - 1) * 2**bits / exp_partial_sum).to(torch.int32)
        diff = global_max.unsqueeze(-1) - x_3d
        shift = torch.floor(diff * eps_max + 0.5).to(torch.int32)
        result_3d = torch.floor(exp_partial_sum_inverse.unsqueeze(-1) / (2**shift)).to(torch.uint8)
        
        return result_3d.view(B, n_heads, seq_length, seq_len_kv)

# --------------------------------------
# Standard Transformer Building Blocks
# --------------------------------------

class OverlapPatchMerging(nn.Module):
    """
    An implementation of overlap patch merging that uses a Conv2d to create
    tokens and F.interpolate for robust downsampling.
    """
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, output_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.output_size = output_size

    def forward(self, x):
        x = self.conv(x)
        # Explicitly downsample to the target size for ONNX compatibility
        x = F.interpolate(x, size=self.output_size, mode='bilinear')
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ITASelfAttention(nn.Module):
    """Self-Attention block for standard QAT training."""
    def __init__(self, embed_dim, proj_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        
        self.matmul_qk = torch.Tensor.matmul
        self.matmul_av = torch.Tensor.matmul
        
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
        
        ita_softmax_qconfig = QConfig(
            activation=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0, quant_max=255, dtype=torch.quint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            ),
            weight=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            )
        )
        
        self.dequant_in_softmax = DeQuantStub(ita_symmetric_qconfig)
        
        self.quant_softmax = QuantStub(ita_softmax_qconfig)
        
        self.dequant_av = DeQuantStub(ita_symmetric_qconfig)
        self.quant_av = QuantStub(ita_symmetric_qconfig)
        
        
        
        # For QAT, we use a standard Softmax. This will be replaced by our
        # custom ITASoftmax during the conversion/export step.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        
        # Standard forward pass. QAT stubs handle quantization.
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q_r = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use the functional wrappers and perform the transpose manually
        K_t = K_r.transpose(-2, -1)
        # Use .matmul() for matrix multiplication, NOT .mul()
        attn_logits = self.matmul_qk(Q_r, K_t)
        
        dq_attn_logits = self.dequant_in_softmax(attn_logits)
        
        attn_weights = self.softmax(dq_attn_logits)
        
        q_attn_weights = self.quant_softmax(attn_weights)
        
        # Use .matmul() here as well
        context = self.matmul_av(q_attn_weights, V_r)
        
        context = self.dequant_av(context)
        context = self.quant_av(context)
        
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        output = self.out_proj(context)
        
        return output


class ITAFeedForward(nn.Module):
    """Feed-Forward Network for standard QAT training."""
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        # Use nn.ReLU for QAT as it's a standard fusible operation.
        # It will be replaced with our ITAGELU in the golden model simulation.
        self.activation = nn.ReLU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x