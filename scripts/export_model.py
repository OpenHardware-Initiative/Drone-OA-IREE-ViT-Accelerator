# Save this file as "export_model.py" and run "python export_model.py"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import os

# Attempt to import IREE packages
try:
    import iree.turbine.aot as aot
except ImportError:
    print("IREE packages not found. Please install them:")
    print("pip install iree-turbine")
    exit()

# --- Helper Functions and Placeholders ---

def refine_inputs(X):
    """
    Placeholder for user's input refinement function.
    Replace this with your actual implementation.
    """
    return X

def ita_partial_max(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Emulates ITAPartialMax by applying softmax to only the top-k elements.
    """
    seq_len = logits.size(-1)
    k = min(k, seq_len)
    topk_vals, topk_indices = torch.topk(logits, k, dim=-1)
    mask = torch.zeros_like(logits).scatter(-1, topk_indices, 1.0)
    masked_logits = logits * mask
    weights = F.softmax(masked_logits, dim=-1)
    return weights

# --- Model Definitions ---

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        self.layerNorm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        x = self.cn1(patches)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layerNorm(x)
        return x, H, W

class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."
        self.heads = num_heads
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x1 = x.clone().permute(0, 2, 1)
        x1 = x1.reshape(B, C, H, W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        x1 = self.ln1(x1)
        keyVal = self.keyValueExtractor(x1)
        keyVal = keyVal.reshape(B, -1, 2, self.heads, int(C / self.heads)).permute(2, 0, 3, 1, 4).contiguous()
        k, v = keyVal[0], keyVal[1]
        q = self.query(x).reshape(B, N, self.heads, int(C / self.heads)).permute(0, 2, 1, 3).contiguous()
        dimHead = (C / self.heads)**0.5
        attention = self.smax(q @ k.transpose(-2, -1) / dimHead)
        attention = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.finalLayer(attention)
        return x

class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        self.expanded_channels = channels * expansion_factor
        self.mlp1 = nn.Linear(channels, self.expanded_channels)
        self.depthwise = nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=3, padding='same', groups=self.expanded_channels)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(self.expanded_channels, channels)

    def forward(self, x, H, W):
        x = self.mlp1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, self.expanded_channels, H, W)
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1, 2))
        x = self.mlp2(x)
        return x

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding,
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        B, C, H, W = x.shape
        x, H, W = self.patchMerge(x)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W)
            x = x + self._ffn[i].forward(x, H, W)
            x = self._lNorm[i].forward(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class MultiheadITAWithRequant(nn.Module):
    def __init__(self, embed_dim, num_heads, params=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert params is not None, "Parameters for requantization must be provided"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.params = params

    def requant_shift(self, x, mult, shift):
        x = x * mult
        x = torch.div(x, 2**shift, rounding_mode='floor')
        return torch.clamp(x + self.params["zp"], -128, 127).to(torch.int8)

    def forward(self, q_input, kv_input):
        B_q, N_q, _ = q_input.shape
        B_kv, N_kv, _ = kv_input.shape
        Q = self.q_proj(q_input).reshape(B_q, N_q, self.num_heads, self.head_dim)
        K = self.k_proj(kv_input).reshape(B_kv, N_kv, self.num_heads, self.head_dim)
        V = self.v_proj(kv_input).reshape(B_kv, N_kv, self.num_heads, self.head_dim)
        Q = self.requant_shift(Q.to(torch.int32), self.params["mq"], self.params["sq"])
        K = self.requant_shift(K.to(torch.int32), self.params["mk"], self.params["sk"])
        V = self.requant_shift(V.to(torch.int32), self.params["mv"], self.params["sv"])
        Q = Q.permute(0, 2, 1, 3).to(torch.float32)
        K = K.permute(0, 2, 1, 3).to(torch.float32)
        V = V.permute(0, 2, 1, 3)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = self.requant_shift(attn_logits, self.params["ma"], self.params["sa"])
        attn_weights = ita_partial_max(attn_logits.float(), k=8)
        context = torch.matmul(attn_weights, V.to(torch.float32))
        context = self.requant_shift(context, self.params["mav"], self.params["sav"])
        context = context.permute(0, 2, 1, 3).reshape(B_q, N_q, self.embed_dim)
        output = self.out_proj(context.to(torch.float32))
        output = self.requant_shift(output.to(torch.int32), self.params["mo"], self.params["so"])
        final = self.requant_shift(output.to(torch.int32), self.params["mf"], self.params["sf"])
        return final

class ITASelfAttentionWrapper(nn.Module):
    def __init__(self, channels, embed_dim, num_heads, reduction_ratio, efficient_attn, itaparameters):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        self.self_attn = MultiheadITAWithRequant(embed_dim=embed_dim, num_heads=num_heads, params=itaparameters)
        self.efficient_attn = efficient_attn

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.efficient_attn:
            x1 = x.permute(0, 2, 1).reshape(B, C, H, W)
            x1 = self.cn1(x1)
            x1 = x1.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x1 = self.ln1(x1)
            out = self.self_attn(x, x1)
        else:
            out = self.self_attn(x, x)
        return out.float()

class MiXITAEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding,
                 n_layers, reduction_ratio, num_heads, expansion_factor, embed_dim, efficient_attn=True, itaparameters=None):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self._attn = nn.ModuleList([ITASelfAttentionWrapper(channels=out_channels, embed_dim=embed_dim, num_heads=num_heads,
                                                            reduction_ratio=reduction_ratio, efficient_attn=efficient_attn,
                                                            itaparameters=itaparameters) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self._lNorms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        B, C, H, W = x.shape
        x, H, W = self.patchMerge(x)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W)
            x = x + self._ffn[i].forward(x, H, W)
            x = self._lNorms[i].forward(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

class ITALSTM(nn.Module):
    def __init__(self, itaparameters=None, efficient_attn=True):
        super().__init__()
        if itaparameters is None:
            itaparameters = {"mq": 1.0, "sq": 0, "mk": 1.0, "sk": 0, "mv": 1.0, "sv": 0,
                             "ma": 1.0, "sa": 0, "mav": 1.0, "sav": 0, "mo": 1.0, "so": 0,
                             "mf": 1.0, "sf": 0, "zp": 0}
        self.encoder_blocks = nn.ModuleList([
            MiXITAEncoderLayer(1, 32, 7, 4, 3, 2, 8, 1, 8, 32, efficient_attn, itaparameters),
            MiXITAEncoderLayer(32, 64, 3, 2, 1, 2, 4, 2, 8, 64, efficient_attn, itaparameters)])
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        self.up_sample = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)

    def _encode(self, x):
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return embeds[1:]

    def _decode(self, encoded_features):
        out = torch.cat([self.pxShuffle(encoded_features[1]), self.up_sample(encoded_features[0])], dim=1)
        out = self.down_sample(out)
        return self.decoder(out.flatten(1))

    def forward(self, X):
        X = refine_inputs(X)
        x = X[0]
        encoded_features = self._encode(x)
        out = self._decode(encoded_features)
        out = torch.cat([out, X[1].squeeze(0) / 10, X[2].squeeze(0)], dim=1).float()
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

# --- Dummy Input Generation ---
def generate_dummy_input(traj_len=10):
    depth_images = torch.randn(traj_len, 1, 60, 90)
    control_input = torch.rand(traj_len, 1)
    orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * traj_len).float()
    return [depth_images, control_input, orientation]

def main():
    # --- Instantiate Model and Dummies ---
    torch.manual_seed(0)
    ita_lstm_module = ITALSTM()
    ita_lstm_module.eval()

    # Generate dummy inputs to get shapes for abstract tensors
    dummy_input_initial = generate_dummy_input(traj_len=10)
    with torch.no_grad():
        _, (dummy_h_n, dummy_c_n) = ita_lstm_module(dummy_input_initial)

    # --- Define the Compiled Module API for IREE ---
    class CompiledITALSTM(aot.CompiledModule):
        params = aot.export_parameters(ita_lstm_module, mutable=True)
        compute = aot.jittable(ita_lstm_module.forward)

        def main(self,
                 depth_images=aot.abstractify(dummy_input_initial[0]),
                 control_input=aot.abstractify(dummy_input_initial[1]),
                 orientation=aot.abstractify(dummy_input_initial[2])):
            return self.compute([depth_images, control_input, orientation])

        def run_with_state(self,
                           depth_images=aot.abstractify(dummy_input_initial[0]),
                           control_input=aot.abstractify(dummy_input_initial[1]),
                           orientation=aot.abstractify(dummy_input_initial[2]),
                           h_n=aot.abstractify(dummy_h_n),
                           c_n=aot.abstractify(dummy_c_n)):
            hidden_state = (h_n, c_n)
            return self.compute([depth_images, control_input, orientation, hidden_state])

    # --- AOT Export ---
    print("Exporting the model with iree.turbine.aot...")
    # The aot.export function converts the PyTorch model into an MLIR module.
    export_output = aot.export(CompiledITALSTM)

    # Define file paths
    output_dir = "compiled_output"
    mlir_file_path = os.path.join(output_dir, "italstm_pytorch_vanilla.mlir")
    vmfb_file_path = os.path.join(output_dir, "italstm_llvm_cpu.vmfb")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the exported MLIR to a file.
    export_output.save_mlir(mlir_file_path)
    
    print("\n✅ Successfully exported MLIR artifact!")
    print(f"   -> {mlir_file_path}")
    
    print("\nNext step: Compile the MLIR file using the command line.")
    print("-" * 60)
    print("Compile for a generic CPU target by running this command:")
    print(f"iree-compile --iree-input-type=torch --iree-hal-target-backend=llvm-cpu \\\n"
          f"  {mlir_file_path} \\\n"
          f"  -o {vmfb_file_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()