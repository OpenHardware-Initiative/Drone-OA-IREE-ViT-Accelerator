# ita_model_export.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# It is assumed that the following modules are available in your project structure.
# Make sure the paths are correct for your environment.
from models.ITA.QAT.layers import OverlapPatchMerging
from models.ITA_single_layer_upsample_shuffle.model import refine_inputs, ITALSTMNetVIT

class DummyHardwareBlock(nn.Module):
    """
    A dummy module to placeholder hardware-accelerated operations.
    This module implements the requested x = x + x operation and is used
    to replace both the ITASelfAttention and ITAFeedForward blocks.
    It contains no trainable parameters.
    """
    def __init__(self):
        super().__init__()
        # This module is intentionally empty as it only performs a stateless operation.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the placeholder operation. The tensor shape remains unchanged.
        In: [B, N, C], Out: [B, N, C]
        """
        # As requested, the replacement operation is x = x + x
        return x + x

class ITAForONNXExport(nn.Module):
    """
    A float-only version of the ITALSTMNetVIT_QAT model prepared for ONNX export.

    In this version, the custom hardware-accelerated blocks (ITASelfAttention_QAT,
    ITAFeedForward_QAT) are replaced by DummyHardwareBlock modules. All other
    layers, which are intended to run on the CPU, are preserved as standard
    torch.nn float modules.
    """
    def __init__(self, num_layers: int = 1):
        super().__init__()
        
        self.num_layers = num_layers
        # Match dimensions from the original ITALSTMNetVIT_QAT model
        self.E = 64
        
        # --- 1. CPU Pre-processing (Float) ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. Transformer Blocks (Replaced with Dummies for ONNX) ---
        # These ModuleLists use the *exact same attribute names* as in the
        # original model ('attention_blocks', 'ffn_blocks', 'norms1', 'norms2')
        # to facilitate weight loading with strict=False.
        self.attention_blocks = nn.ModuleList([DummyHardwareBlock() for _ in range(num_layers)])
        self.ffn_blocks = nn.ModuleList([DummyHardwareBlock() for _ in range(num_layers)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(num_layers)])
        
        # --- 3. CPU Feature Fusion (Float) ---
        # This section is preserved identically from the original model.
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.up_sample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        self.down_sample = nn.Conv2d(in_channels=(self.E // 4) + self.E, out_channels=9, kernel_size=3, padding=1)
        
        # --- 4. CPU Post-processing (Float) ---
        self.decoder = nn.Linear(4608, 512)
        # Note: LSTM dropout is automatically disabled in eval() mode.
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3)
        self.nn_fc2 = nn.Linear(128, 3)

    def load_float_weights_from_trained_model(self, trained_model: ITALSTMNetVIT):
        """
        Copies weights from a trained QAT model to this export-ready model.

        It uses `load_state_dict` with `strict=False`, which correctly loads weights
        for all layers with matching names (tokenizer, norms, decoder, etc.) and
        skips the layers that don't exist in this model (quantizers, dequantizers)
        or have no parameters (the dummy blocks).
        """
        trained_state_dict = trained_model.state_dict()
        self.load_state_dict(trained_state_dict, strict=False)
        print("✅ Successfully loaded weights for all corresponding float modules.")

    def forward(self, X):
        """
        Defines the forward pass, mirroring the QAT model's structure but without
        any quantization/dequantization steps.
        """
        # Input refinement is identical to the training model
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        # --- 1. Tokenizer ---
        x, H, W = self.tokenizer(img_data)
        
        # --- 2. Loop through the transformer layers ---
        for i in range(self.num_layers):
            # Attention sub-block (with dummy operation)
            attn_out = self.attention_blocks[i](x)
            x = x + attn_out  # Residual connection
            x = self.norms1[i](x)

            # FFN sub-block (with dummy operation)
            ffn_out = self.ffn_blocks[i](x)
            x = x + ffn_out  # Residual connection
            x = self.norms2[i](x)
        
        # --- 3. Feature Fusion ---
        B, S, E = x.shape
        x_2d = x.transpose(1, 2).view(B, E, H, W)
        x_shuffled = self.pxShuffle(x_2d)
        x_upsampled = self.up_sample(x_2d)
        x_fused = torch.cat([x_shuffled, x_upsampled], dim=1)
        x_down = self.down_sample(x_fused)

        # --- 4. Decoder and LSTM ---
        x = x_down.flatten(1)
        out = self.decoder(x)
        
        # Use torch.cat directly for float models
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h