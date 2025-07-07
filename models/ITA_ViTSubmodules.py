import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadITAWithRequant(nn.Module):
    def __init__(self, embed_dim, num_heads, params=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert params is not None, "Parameters for requantization must be provided"

        # Projections: shared across heads but produce concatenated head_dim output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Requant parameters
        self.params = params

    def requant_shift(self, x, mult, shift):
        x = x * mult
        x = torch.div(x, 2 ** shift, rounding_mode='floor')
        return torch.clamp(x + self.params["zp"], -128, 127).to(torch.int8)

    def forward(self, q_input, kv_input):
        B_q, N_q, _ = q_input.shape
        B_kv, N_kv, _ = kv_input.shape

        # Linear projections
        Q = self.q_proj(q_input).reshape(B_q, N_q, self.num_heads, self.head_dim)
        K = self.k_proj(kv_input).reshape(B_kv, N_kv, self.num_heads, self.head_dim)
        V = self.v_proj(kv_input).reshape(B_kv, N_kv, self.num_heads, self.head_dim)

        Q = self.requant_shift(Q.to(torch.int32), self.params["mq"], self.params["sq"])
        K = self.requant_shift(K.to(torch.int32), self.params["mk"], self.params["sk"])
        V = self.requant_shift(V.to(torch.int32), self.params["mv"], self.params["sv"])

        # Attention computation per head
        Q = Q.permute(0, 2, 1, 3)  # (B, H, N, D)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            Q = Q.to(torch.float32)
            K = K.to(torch.float32)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, N, N)
        attn_logits = self.requant_shift(attn_logits, self.params["ma"], self.params["sa"])

        attn_weights = ita_partial_max(attn_logits.float(), k=8)

        context = torch.matmul(attn_weights, V.to(torch.float32))  # (B, H, N, D)
        context = self.requant_shift(context, self.params["mav"], self.params["sav"])

        # Concatenate all heads
        context = context.permute(0, 2, 1, 3).reshape(B_q, N_q, self.embed_dim)

        output = self.out_proj(context.to(torch.float32))  # Use float proj for now
        output = self.requant_shift(output.to(torch.int32), self.params["mo"], self.params["so"])
        final = self.requant_shift(output.to(torch.int32), self.params["mf"], self.params["sf"])

        return final

def ita_partial_max(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Emulates ITAPartialMax by applying softmax to only the top-k elements along the last axis.
    If k exceeds the dimension size, it is clipped.
    """
    seq_len = logits.size(-1)
    k = min(k, seq_len)  # Prevent topk from throwing
    topk_vals, topk_indices = torch.topk(logits, k, dim=-1)
    mask = torch.zeros_like(logits).scatter(-1, topk_indices, 1.0)
    masked_logits = logits * mask
    weights = F.softmax(masked_logits, dim=-1)
    return weights

class ITASelfAttentionWrapper(nn.Module):
    def __init__(self, channels, embed_dim, num_heads, reduction_ratio, efficient_attn, itaparameters):
        super().__init__()
        # Reduction Parameters #
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride= reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Parameters #
        self.self_attn = MultiheadITAWithRequant(embed_dim=embed_dim, num_heads=num_heads, params=itaparameters)
        self.efficient_attn = efficient_attn

    def forward(self, x, H, W):
        B,N,C = x.shape
        # B, N, C -> B, C, N
        # Optional spatial reduction for keys and values
        if self.efficient_attn:
            x1 = x.permute(0,2,1)
            # BCN -> BCHW
            x1 = x1.reshape(B,C,H,W)
            x1 = self.cn1(x1)
            x1 = x1.reshape(B,C,-1)
            x1 = x1.permute(0,2,1).contiguous()
            x1 = self.ln1(x1)

            # Perform attention with (x as query, x1 as kv)
            out = self.self_attn(x, x1)

        else:
            # Perform attention with (x as query and kv)
            out = self.self_attn(x, x)
        return out.float()
    
class MiXITAEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor, embed_dim, efficient_attn=True, itaparameters=None):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        self._attn = nn.ModuleList([ITASelfAttentionWrapper(channels=out_channels,
                                                            embed_dim=embed_dim, 
                                                            num_heads=num_heads, 
                                                            reduction_ratio=reduction_ratio, 
                                                            efficient_attn=efficient_attn, 
                                                            itaparameters=itaparameters
                                                            ) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])
    
    def forward(self, x):
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x) # B N embed dim (C)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W) #BNC
            x = x + self._ffn[i].forward(x, H, W) #BNC  # Skip connections
            x = self._lNorms[i].forward(x) #BNC
        # Reshape tokens back to spatial format for next stage: (B, N, C) → (B, C, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x