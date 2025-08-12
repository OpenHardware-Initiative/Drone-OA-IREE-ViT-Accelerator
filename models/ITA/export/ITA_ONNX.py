# ita_model_export.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from collections import OrderedDict

# Import the float modules from your original files
from models.ITA.QAT.layers import OverlapPatchMerging 
# (refine_inputs function is also needed here)
from models.ITA.QAT.model import refine_inputs, ITALSTMNetVIT_QAT


class DummyITASelfAttention(nn.Module):
    """A dummy module that mimics the output shape of ITASelfAttention."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # The attention block does not change the tensor shape
        # In: [B, N, C], Out: [B, N, C]
        # We return random data to signify this part is handled by hardware.
        return torch.randn_like(x)

class DummyITAFeedForward(nn.Module):
    """A dummy module that mimics the output shape of ITAFeedForward."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # The FFN block also does not change the tensor shape
        # In: [B, N, C], Out: [B, N, C]
        return torch.randn_like(x)


class ITAForONNXExport(nn.Module):
    """
    A version of the model for ONNX export, with quantized parts
    replaced by dummy modules.
    """
    def __init__(self):
        super().__init__()
        
        self.E, self.S, self.P, self.F, self.H = 128, 128, 192, 256, 1
        
        # --- 1. CPU Pre-processing (Float) ---
        self.tokenizer = OverlapPatchMerging(
            in_channels=1, out_channels=self.E, patch_size=7, 
            stride=2, padding=3, output_size=(8, 16)
        )

        # --- 2. Dummy Placeholders for ITA Hardware ---
        self.attention_blocks = nn.ModuleList([DummyITASelfAttention() for _ in range(2)])
        self.ffn_blocks = nn.ModuleList([DummyITAFeedForward() for _ in range(2)])
        
        # --- CPU-bound Layers (Float) ---
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(self.E) for _ in range(2)])
        
        # --- 3. CPU Post-processing (Float) ---
        self.decoder = nn.Linear(self.E * self.S, 512)
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3) # dropout is off for inference
        self.nn_fc2 = nn.Linear(128, 3)
        
        # Use standard FloatFunctional for ONNX export
        self.add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()

    def load_float_weights_from_trained_model(self, trained_model: ITALSTMNetVIT_QAT):
        """
        Copies weights from the trained QAT model to this export model for
        all corresponding float modules.
        """
        # Create a mapping of module names in this model to the trained model
        # Note: ModuleList names are the same.
        float_modules_to_copy = [
            "tokenizer", "norm1_layers", "norm2_layers", "decoder", "lstm", "nn_fc2"
        ]
        
        # Load the state dict from the CPU-version of the trained model
        trained_state_dict = trained_model.state_dict()
        self.load_state_dict(trained_state_dict, strict=False)

        print("Successfully loaded weights for float modules.")


    def forward(self, X):
        # The forward pass is identical in structure to the training model
        X = refine_inputs(X)
        img_data, additional_data, quat_data = X[0], X[1], X[2]

        x, H, W = self.tokenizer(img_data)
        
        for i in range(len(self.attention_blocks)):
            attn_out = self.attention_blocks[i](x)
            x = self.add.add(x, attn_out)
            x = self.norm1_layers[i](x)

            ffn_out = self.ffn_blocks[i](x)
            x = self.add.add(x, ffn_out)
            x = self.norm2_layers[i](x)
        
        x = x.flatten(1)
        out = self.decoder(x)
        out_cat = self.cat.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        hidden_state = X[3] if len(X) > 3 else None
        out_lstm, h = self.lstm(out_cat, hidden_state)
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        return out_final, h