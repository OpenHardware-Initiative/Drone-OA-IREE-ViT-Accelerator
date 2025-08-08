# models/testing/export/ITA_layers_export.py

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional

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

class ITASelfAttention_Export(nn.Module):
    """
    The final, export-ready Self-Attention block.
    
    This version uses the hardware-compliant ITASoftmax module and is structured
    to return all intermediate integer tensors for hardware verification.
    """
    def __init__(self, embed_dim, proj_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads

        # Standard quantized linear layers
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)

        # QAT-compatible wrappers for matrix multiplication
        self.matmul_qk = FloatFunctional()
        self.matmul_av = FloatFunctional()

        # Use the custom ITASoftmax for the export model
        self.softmax = ITASoftmax()

    def forward(self, x, H, W):
        # This dictionary will store the intermediate tensors
        intermediates = {}
        B, N, _ = x.shape

        # --- Projections ---
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Store the integer representation of projection outputs
        intermediates['Qp_requant'] = Q.int_repr()
        intermediates['Kp_requant'] = K.int_repr()
        intermediates['Vp_requant'] = V.int_repr()

        # --- Attention Score Calculation ---
        Q_r = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        K_t = K_r.transpose(-2, -1)
        attn_logits = self.matmul_qk.matmul(Q_r, K_t)
        intermediates['A_requant'] = attn_logits.int_repr()

        # --- Custom Softmax and Requantization ---
        attn_logits_int = attn_logits.int_repr()
        attn_weights_int = self.softmax(attn_logits_int)
        intermediates['A_partial_softmax'] = attn_weights_int
        
        attn_weights_q = torch.quantize_per_tensor(
            attn_weights_int.float(),
            attn_logits.q_scale(),
            attn_logits.q_zero_point(),
            torch.qint8
        )
        
        # --- Context Vector Calculation ---
        context = self.matmul_av.matmul(attn_weights_q, V_r)
        intermediates['O_soft_requant'] = context.int_repr()
        
        # --- Output Projection ---
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        output = self.out_proj(context)
        intermediates['Out_soft_requant'] = output.int_repr()
        
        return output, intermediates


class ITAFeedForward_Export(nn.Module):
    """
    Export-ready Feed-Forward Network that also returns intermediates.
    """
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        # In the converted model, this will be a quantized ReLU.
        self.activation = nn.ReLU()

    def forward(self, x, H, W):
        intermediates = {}

        x1 = self.fc1(x)
        intermediates['FFp_requant'] = x1.int_repr()
        
        x_act = self.activation(x1)
        # Note: The 'gelu' key is used to match the golden model script,
        # even though we are using ReLU in QAT.
        intermediates['relu'] = x_act.int_repr()
        
        x2 = self.fc2(x_act)
        intermediates['FF2p_requant'] = x2.int_repr()
        
        return x2, intermediates