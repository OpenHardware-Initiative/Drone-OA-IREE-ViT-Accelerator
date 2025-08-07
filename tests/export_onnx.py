import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
import sys

# Import the necessary torch.quantization components
from torch.quantization import QConfig, FusedMovingAvgObsFakeQuantize, prepare_qat, convert

# --- Add project root to path to import model files ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.ITA.ITA_model import ITALSTMNetVIT
from models.ITA.export.ITA_model_export import ITALSTMNetVIT_Export
from models.ITA.dispatch.ITA_model_placeholder import ITALSTMNetVIT_Placeholder_Export

# ==============================================================================
#  1. ONNX-FRIENDLY FORWARD PASS (No changes here)
# ==============================================================================
def forward_onnx(self, img_data, additional_data, quat_data, h_in, c_in):
    """
    An ONNX-compatible forward method that accepts a flat tuple of tensors.
    """
    # The conditional shape check is removed as we provide a correctly-sized dummy input.
    # if img_data.shape[-2:] != (60, 90):
    #     img_data = nn.functional.interpolate(img_data, size=(60, 90), mode='bilinear', align_corners=False)

    hidden_state = (h_in, c_in)
    x_float, H, W = self.tokenizer(img_data)

    for i in range(len(self.attention_blocks)):
        res_attn = x_float
        x_quant_attn = self.quant_attention[i](x_float)

        # --- FIX 1: Unpack the tuple returned by the attention block ---
        # The block returns (output_tensor, intermediates). We only need the tensor.
        x_quant_attn_out, _ = self.attention_blocks[i](x_quant_attn, H, W)

        x_float = self.dequant_attention[i](x_quant_attn_out)
        x_float = res_attn + x_float
        x_float = self.norm1_layers[i](x_float)

        res_ffn = x_float
        x_quant_ffn = self.quant_ffn[i](x_float)

        # --- FIX 2: Unpack the tuple returned by the FFN block ---
        x_quant_ffn_out, _ = self.ffn_blocks[i](x_quant_ffn, H, W)
        
        x_float = self.dequant_ffn[i](x_quant_ffn_out)
        x_float = res_ffn + x_float
        x_float = self.norm2_layers[i](x_float)

    x = x_float.flatten(1)
    # 1. Use the new stub to convert the float tensor 'x' to a quantized tensor
    x_quant_decoder = self.quant_decoder(x)
    # 2. Feed the new quantized tensor into the decoder
    out = self.decoder(x_quant_decoder)
    out_float = self.dequant_out(out)
    out_cat = torch.cat([out_float, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
    out_lstm, (h_out, c_out) = self.lstm(out_cat, hidden_state)
    out_final = self.nn_fc2(out_lstm.squeeze(0))

    return out_final, h_out, c_out

# ==============================================================================
#  2. MAIN EXPORT LOGIC
# ==============================================================================
def main():
    print("🚀 Starting ONNX export process for ITALSTMNetVIT...")

    # --- Step 1: Instantiate the Model ---
    model = ITALSTMNetVIT_Export()
    
    # --- Step 2: Replicate the Full Quantization Conversion Process ---
    print("🛠️  Preparing model architecture for quantized weights...")
    
    torch.backends.quantized.engine = 'qnnpack'
    
    ita_symmetric_qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False
        ),
        weight=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False
        )
    )
    
    model.qconfig = ita_symmetric_qconfig
    model.tokenizer.qconfig = None
    for block in model.norm1_layers:
        block.qconfig = None
    for block in model.norm2_layers:
        block.qconfig = None
    
    
    
    # **FINAL FIX**: Switch to train() mode for prepare_qat, then back to eval() for convert.
    model.train()
    prepare_qat(model, inplace=True)
    model.eval()
    convert(model, inplace=True)

    print("✅ Model architecture correctly converted to quantized form.")
    
    # --- Step 3: Load the FINAL Converted Quantized State ---
    quantized_model_path = "/Projects/Agus/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t11_02_qat_rom_scratch/model_quantized_final.pth"
    if not os.path.exists(quantized_model_path):
        print(f"❌ Error: Model file not found at {quantized_model_path}")
        return

    print(f"🔍 Loading quantized state_dict from: {quantized_model_path}")
    state_dict = torch.load(quantized_model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # --- Step 4: Prepare Model for Export ---
    model.forward = forward_onnx.__get__(model, ITALSTMNetVIT)
    print("✅ Patched model's forward method for ONNX compatibility.")

    # --- Step 5: Create Dummy Inputs ---
    batch_size = 1
    dummy_img = torch.randn(batch_size, 1, 60, 90, dtype=torch.float32)
    dummy_add = torch.randn(batch_size, 1, dtype=torch.float32)
    dummy_quat = torch.randn(batch_size, 4, dtype=torch.float32)
    dummy_h_in = torch.randn(3, batch_size, 128, dtype=torch.float32)
    dummy_c_in = torch.randn(3, batch_size, 128, dtype=torch.float32)
    dummy_inputs = (dummy_img, dummy_add, dummy_quat, dummy_h_in, dummy_c_in)
    print("✅ Created dummy inputs for tracing.")

    # --- Step 6: Configure and Run ONNX Export ---
    onnx_output_path = "ita_model_quantized.onnx"
    input_names = ["image", "additional_data", "quat_data", "h_in", "c_in"]
    output_names = ["final_output", "h_out", "c_out"]
    dynamic_axes = {
        'image': {0: 'batch_size'}, 'additional_data': {0: 'batch_size'},
        'quat_data': {0: 'batch_size'}, 'h_in': {1: 'batch_size'},
        'c_in': {1: 'batch_size'}, 'final_output': {0: 'batch_size'},
        'h_out': {1: 'batch_size'}, 'c_out': {1: 'batch_size'},
    }

    print(f"📦 Exporting model to ONNX at: {onnx_output_path}")
    torch.onnx.export(
        model, dummy_inputs, onnx_output_path,
        input_names=input_names, output_names=output_names,
        opset_version=17, dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("✅ Export complete.")

    # --- Step 7: Verify and Test the ONNX Model ---
    print("🔬 Verifying and testing the exported ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_output_path)
    ort_inputs = {name: tensor.numpy() for name, tensor in zip(input_names, dummy_inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("✅ ONNX Runtime inference test successful.")
    print(f"   - Output shape: {ort_outs[0].shape}")
    print("\n🎉 Successfully exported and verified the quantized model!")


if __name__ == "__main__":
    main()