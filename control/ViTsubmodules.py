"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the submodules for ViT that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al

@source: https://github.com/git-dhruv/Segformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride = stride, padding = padding)
        self.layerNorm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        """Merge patches to reduce dimensions of input.

        :param patches: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        x = self.cn1(patches)
        _,_,H, W = x.shape
        x = x.flatten(2).transpose(1,2) #Flatten - (B,C,H*W); transpose B,HW, C
        x = self.layerNorm(x)
        return x,H,W #B, N, EmbedDim
class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.heads= num_heads

        #### Self Attention Block consists of 2 parts - Reduction and then normal Attention equation of queries and keys###
        
        # Reduction Parameters #
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride= reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Parameters #
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels) 


    def forward(self, x, H, W):

        """ Perform self attention with reduced sequence length

        :param x: tensor of shape (B, N, C) where
            B is the batch size,
            N is the number of queries (equal to H * W)
            C is the number of channels
        :return: tensor of shape (B, N, C)
        """
        B,N,C = x.shape
        # B, N, C -> B, C, N
        x1 = x.clone().permute(0,2,1)
        # BCN -> BCHW
        x1 = x1.reshape(B,C,H,W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B,C,-1).permute(0,2,1).contiguous()
        x1 = self.ln1(x1)
        # We have got the Reduced Embeddings! We need to extract key and value pairs now
        keyVal = self.keyValueExtractor(x1)
        keyVal = keyVal.reshape(B, -1 , 2, self.heads, int(C/self.heads)).permute(2,0,3,1,4).contiguous()
        k,v = keyVal[0],keyVal[1] #b,heads, n, c/heads
        q = self.query(x).reshape(B, N, self.heads, int(C/self.heads)).permute(0, 2, 1, 3).contiguous()

        dimHead = (C/self.heads)**0.5
        attention = self.smax(q@k.transpose(-2, -1)/dimHead)
        attention = (attention@v).transpose(1,2).reshape(B,N,C)

        x = self.finalLayer(attention) #B,N,C        
        return x

class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels*expansion_factor
        #MLP Layer        
        self.mlp1 = nn.Linear(channels, expanded_channels)
        #Depth Wise CNN Layer
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.gelu = nn.GELU()
        #MLP to predict
        self.mlp2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        """ Perform self attention with reduced sequence length

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        # Input BNC instead of BCHW
        # BNC -> B,N,C*exp 
        x = self.mlp1(x)
        B,N,C = x.shape
        # Prepare for the CNN operation, channel should be 1st dim
        # B,N, C*exp -> B, C*exp, H, W 
        x = x.transpose(1,2).view(B,C,H,W)

        #Depth Conv - B, N, Cexp 
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1,2))

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        """ Run one block of the mix vision transformer

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x) # B N embed dim (C)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W) #BNC
            x = x + self._ffn[i].forward(x, H, W) #BNC
            x = self._lNorm[i].forward(x) #BNC
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x

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
