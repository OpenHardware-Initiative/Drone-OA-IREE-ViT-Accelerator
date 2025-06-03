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



class ITA(nn.Module):
    """
    Integer Transformer Accelerator (ITA) attention block for deployment.
    This version assumes quantized weights and executes pure attention logic.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, q_input, kv_input):
        """
        Emulates the ONNX ITA graph structure:
        - Linear projections for Q, K, V
        - QKᵀ attention with softmax approximation (ITAPartialMax)
        - Attention output projection

        Parameters:
        - q_input: (B, N, C), e.g. query from patch embeddings
        - kv_input: (B, N, C), same or different source for key/value

        Returns:
        - (B, N, C): processed token features
        """
        B, N, C = q_input.shape

        # Linear projections (assumed quantized on hardware)
        Q = self.q_proj(q_input)
        K = self.k_proj(kv_input)
        V = self.v_proj(kv_input)

        # Attention mechanism
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, N, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # ITAPartialMax equivalent
        context = torch.matmul(attn_weights, V)             # (B, N, C)

        # Final output projection
        output = self.out_proj(context)
        return output


class ITAWithRequant(nn.Module):
    """
    High-fidelity emulation of the Integer Transformer Accelerator (ITA) attention block.
    Implements all 7 stages using quantized projections, RequantShift logic, and approximate softmax.
    """

    def __init__(self, embed_dim,
                 multiplier_q=1.0, shift_q=0,
                 multiplier_k=1.0, shift_k=0,
                 multiplier_v=1.0, shift_v=0,
                 multiplier_attn=1.0, shift_attn=0,
                 multiplier_av=1.0, shift_av=0,
                 multiplier_o=1.0, shift_o=0,
                 multiplier_final=1.0, shift_final=0,
                 zero_point=0):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Requantization parameters
        self.mq, self.sq = multiplier_q, shift_q
        self.mk, self.sk = multiplier_k, shift_k
        self.mv, self.sv = multiplier_v, shift_v
        self.ma, self.sa = multiplier_attn, shift_attn
        self.mav, self.sav = multiplier_av, shift_av
        self.mo, self.so = multiplier_o, shift_o
        self.mf, self.sf = multiplier_final, shift_final
        self.zero_point = zero_point

    def requant_shift(self, x, multiplier, shift):
        """Simulate RequantShift: (x * multiplier) >> shift + zero_point, clamped to int8."""
        scaled = x * multiplier
        shifted = torch.div(scaled, 2 ** shift, rounding_mode='floor')
        return torch.clamp(shifted + self.zero_point, -128, 127).to(torch.int8)

    def forward(self, q_input, kv_input):
        """
        Emulates:
        [1–3] Linear projections with quant → [4] attention logits → softmax
        [5] weighted values → [6] output projection → [7] final requant
        """
        B, N, C = q_input.shape

        # Stage 1–3: Linear projections and Requant
        Q = self.requant_shift(self.q_proj(q_input).to(torch.int32), self.mq, self.sq)
        K = self.requant_shift(self.k_proj(kv_input).to(torch.int32), self.mk, self.sk)
        V = self.requant_shift(self.v_proj(kv_input).to(torch.int32), self.mv, self.sv)

        # Stage 4: Q × Kᵗ → Requant → Approx. Softmax
        attn_logits = torch.matmul(Q.to(torch.int32), K.transpose(-2, -1).to(torch.int32))
        attn_logits = self.requant_shift(attn_logits, self.ma, self.sa)
        attn_weights = ita_partial_max(attn_logits.float(), k=8)

        # Stage 5: A × V → Requant
        context = torch.matmul(attn_weights, V.to(torch.int32))
        context = self.requant_shift(context, self.mav, self.sav)

        # Stage 6: Output projection → Requant
        output = self.requant_shift(self.out_proj(context).to(torch.int32), self.mo, self.so)

        # Stage 7: Final scaling (e.g., post-layer normalization or pooling)
        final = self.requant_shift(output.to(torch.int32), self.mf, self.sf)

        return final

def ita_partial_max(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Emulates ITAPartialMax by applying softmax to only the top-k elements along the last axis.
    Remaining entries are masked to zero.
    """
    # logits: (B, N, N)
    topk_vals, topk_indices = torch.topk(logits, k, dim=-1)
    mask = torch.zeros_like(logits).scatter(-1, topk_indices, 1.0)
    masked_logits = logits * mask

    # Apply softmax only over masked values
    weights = F.softmax(masked_logits, dim=-1)
    return weights