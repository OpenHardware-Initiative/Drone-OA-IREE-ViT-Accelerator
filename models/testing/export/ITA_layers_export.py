# ita_layers_export.py

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

# Import the base hardware-compliant softmax
from models.testing.ITA_layers import OverlapPatchMerging, ITASoftmax


# --------------------------------------
# Export-Specific Transformer Blocks
# --------------------------------------

class ITASelfAttention_Export(nn.Module):
    """Self-Attention block for EXPORTING intermediate tensors."""
    def __init__(self, embed_dim, proj_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        
        self.softmax = ITASoftmax()
        
        self.matmul_qk = FloatFunctional()
        self.matmul_av = FloatFunctional()

    def forward(self, x, H, W):
        B, N, _ = x.shape
        intermediates = {}

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        intermediates['Qp_requant'] = Q.int_repr()
        intermediates['Kp_requant'] = K.int_repr()
        intermediates['Vp_requant'] = V.int_repr()

        Q_r = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = self.matmul_qk.mul_transpose(Q_r, K_r)
        intermediates['A_requant'] = attn_logits.int_repr()
        
        attn_weights = self.softmax(attn_logits.int_repr())
        intermediates['A_partial_softmax'] = attn_weights
        
        attn_weights_q = torch.quantize_per_tensor(
            attn_weights.float(), attn_logits.q_scale(), attn_logits.q_zero_point(), torch.quint8
        )
            
        context = self.matmul_av.mul(attn_weights_q, V_r)
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        intermediates['O_soft_requant'] = context.int_repr()

        output = self.out_proj(context)
        intermediates['Out_soft_requant'] = output.int_repr()
            
        return output, intermediates


class ITAFeedForward_Export(nn.Module):
    """FFN for EXPORTING intermediate tensors."""
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        intermediates = {}
        
        ffn1_out = self.fc1(x)
        intermediates['FFp_requant'] = ffn1_out.int_repr()
            
        relu_out = self.relu(ffn1_out)
        # Note: This is the output of PyTorch's ReLU. The golden model will simulate ITAGELU.
        intermediates['relu'] = relu_out.int_repr() 
            
        ffn2_out = self.fc2(relu_out)
        intermediates['FF2p_requant'] = ffn2_out.int_repr()
            
        return ffn2_out, intermediates