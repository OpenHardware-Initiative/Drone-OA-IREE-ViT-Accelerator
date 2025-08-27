# models/ITA/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchMerging(nn.Module):
    """
    Implementation of overlap patch merging.
    This module is already float-only and requires no changes.
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
    
class ITAFeedForward(nn.Module):
    """
    Standard floating-point Feed-Forward Network.
    - Removed QuantStub and DeQuantStub.
    """
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
    def forward(self, x):
        # Direct floating-point operations
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class ITASelfAttention(nn.Module):
    """
    Standard floating-point Self-Attention block.
    - Removed QuantStub and DeQuantStub.
    - Replaced IntegerApproximatedSoftmax with nn.Softmax.
    - Replaced FloatFunctional matmul with torch.matmul.
    """
    def __init__(self, embed_dim, proj_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)
        self.v_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, embed_dim)
        
        # Use the standard, numerically precise softmax for float training
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, _ = x.shape
        
        # Standard linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use standard torch.matmul
        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = self.softmax(attn_logits)
        
        context = torch.matmul(attn_weights, V)
        context = context.permute(0, 2, 1, 3).reshape(B, N, self.q_proj.out_features)
        
        output = self.out_proj(context)
        return output