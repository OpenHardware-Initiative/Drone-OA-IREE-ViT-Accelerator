import torch
import torch.nn as nn
import torch.nn.quantized as nnq

from models.ITA_ViTSubmodules import ITASelfAttentionWrapper

from third_party.vitfly.models.ViTsubmodules import OverlapPatchMerging, EfficientSelfAttention

class QuantReadyTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([QuantReadyMixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

        self.adder = nnq.FloatFunctional()

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

            x = self.adder.add(x, self._attn[i].forward(x, H, W))       #using FloatFunctional for quantization compatibility
            x = self.adder.add(x, self._ffn[i].forward(x, H, W))        #using FloatFunctional for quantization compatibility
            x = self._lNorm[i].forward(x) #BNC
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x
    
class QuantReadyMixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels*expansion_factor
        #MLP Layer        
        self.mlp1 = nn.Linear(channels, expanded_channels)
        #Depth Wise CNN Layer
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.activation = nn.ReLU()
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
        x = self.activation(self.depthwise(x).flatten(2).transpose(1,2))

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x
    
class QuantReadyITAEncoderLayer(nn.Module):
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
        self._ffn = nn.ModuleList([QuantReadyMixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])
        self.adder = nnq.FloatFunctional()
    
    def forward(self, x):
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x) # B N embed dim (C)
        for i in range(len(self._attn)):
            x = self.adder.add(x, self._attn[i].forward(x, H, W))
            x = self.adder.add(x, self._ffn[i].forward(x, H, W)) #BNC  # Skip connections
            x = self._lNorms[i].forward(x) #BNC
        # Reshape tokens back to spatial format for next stage: (B, N, C) → (B, C, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x