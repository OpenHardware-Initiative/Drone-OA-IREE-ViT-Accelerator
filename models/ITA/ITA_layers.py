# ita_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Helper Functions
# --------------------------------------

def requantize(x, mult, shift, add=0):
    """A PyTorch implementation of the ITA hardware's requantization function."""
    x = x * mult
    x = torch.div(x, 2**shift, rounding_mode='floor') + add
    return torch.clamp(x, -128, 127)

# --------------------------------------
# Hardware-Compliant Modules
# --------------------------------------

class ITAGELU(nn.Module):
    """A PyTorch implementation of the integer-approximated GELU activation."""
    def __init__(self, params):
        super().__init__()
        self.q_1 = torch.tensor(params.get("q_1", -22), dtype=torch.int16)
        self.q_b = torch.tensor(params.get("q_b", -14), dtype=torch.int16)
        self.q_c = torch.tensor(params.get("q_c", 24), dtype=torch.int16)
        self.eps_mul = torch.tensor(params.get("gelu_rqs_mul", 119), dtype=torch.uint8)
        self.eps_shift = torch.tensor(params.get("gelu_rqs_shift", 20), dtype=torch.uint8)
        self.eps_add = torch.tensor(params.get("gelu_rqs_add", 0), dtype=torch.uint8)

    def _i_poly(self, q, q_b, q_c):
        d = q.to(torch.int16) + q_b
        return (d * d + q_c.to(torch.int32)).to(torch.int32)

    def _i_erf(self, q, q_b, q_c):
        q_sgn = torch.sign(q)
        q_abs = torch.abs(q)
        q_clipped = torch.clamp(q_abs, 0, -q_b)
        return q_sgn * self._i_poly(q_clipped, q_b, q_c)

    def _i_gelu(self, q, q_1, q_b, q_c):
        q_clipped = torch.clamp(q, -127, 127)
        q_erf = self._i_erf(q_clipped, q_b, q_c)
        return q_clipped * (q_erf + q_1)

    def forward(self, x):
        q_out = self._i_gelu(x, self.q_1, self.q_b, self.q_c)
        return requantize(q_out, self.eps_mul, self.eps_shift, self.eps_add)


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
# Main Transformer Building Blocks
# --------------------------------------

class OverlapPatchMerging(nn.Module):
    """Tokenizer that uses a Conv2d to create tokens."""
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, output_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ITASelfAttention(nn.Module):
    """Hardware-compliant Self-Attention block for training and standard inference."""
    def __init__(self, embed_dim, proj_dim, num_heads, params):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.params = params
        
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        self.softmax = ITASoftmax()

    def forward(self, x, H, W):
        B, N, _ = x.shape
        
        Q = requantize(self.q_proj(x), self.params["mq"], self.params["sq"])
        K = requantize(self.k_proj(x), self.params["mk"], self.params["sk"])
        V = requantize(self.v_proj(x), self.params["mv"], self.params["sv"])

        Q_r = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = requantize(torch.matmul(Q_r.float(), K_r.float().transpose(-2, -1)), self.params["ma"], self.params["sa"])
        attn_weights = self.softmax(attn_logits)
        context = requantize(torch.matmul(attn_weights.float(), V_r.float()), self.params["mav"], self.params["sav"])

        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        output = requantize(self.out_proj(context.float()), self.params["mo"], self.params["so"])
        
        return output


class ITAFeedForward(nn.Module):
    """Hardware-compliant Feed-Forward Network for training and standard inference."""
    def __init__(self, embed_dim, ffn_dim, params):
        super().__init__()
        self.params = params
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = ITAGELU(params) # Or nn.ReLU() depending on hardware

    def forward(self, x, H, W):
        x = requantize(self.fc1(x), self.params["m_ff1"], self.params["s_ff1"])
        x = self.activation(x)
        x = requantize(self.fc2(x), self.params["m_ff2"], self.params["s_ff2"])
        return x