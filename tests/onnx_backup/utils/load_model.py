from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd

class HardcodedQuantizer(nn.Module):
    def __init__(self, scale=1.0, zero_point=0):
        super().__init__()
        self.register_buffer('scale', torch.tensor(float(scale)))
        self.register_buffer('zero_point', torch.tensor(int(zero_point)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.quantize_per_tensor(x, self.scale, self.zero_point, torch.qint8)


def load_weights_to_placeholder_model(original_model: nn.Module, placeholder_model: nn.Module):
    """
    This master function handles the entire transfer process.
    1. DEQUANTIZES weights for standard float layers (LSTM, Linear, etc.).
    2. COPIES scale/zp values for the quantizer stubs at hardware boundaries.
    """
    print("--- Transferring weights and quantization parameters ---")

    # --- Part 1: Load dequantized weights into the float layers ---
    print("\n[INFO] Dequantizing and loading weights for CPU layers...")
    new_state_dict = OrderedDict()
    source_modules = dict(original_model.named_modules())

    for p_name, p_module in placeholder_model.named_modules():
        if p_name in source_modules:
            o_module = source_modules[p_name]

            if isinstance(p_module, nn.Linear) and isinstance(o_module, nn.quantized.Linear):
                print(f"  - Loading weights for Linear layer: '{p_name}'")
                new_state_dict[f"{p_name}.weight"] = o_module.weight().dequantize()
                if o_module.bias() is not None: new_state_dict[f"{p_name}.bias"] = o_module.bias().dequantize()

            elif isinstance(p_module, nn.Conv2d) and isinstance(o_module, nn.quantized.Conv2d):
                print(f"  - Loading weights for Conv2d layer: '{p_name}'")
                new_state_dict[f"{p_name}.weight"] = o_module.weight().dequantize()
                if o_module.bias() is not None: new_state_dict[f"{p_name}.bias"] = o_module.bias().dequantize()

            elif isinstance(p_module, nn.LSTM) and isinstance(o_module, nn.quantized.LSTM):
                print(f"  - Loading weights for LSTM layer: '{p_name}'")
                for i in range(o_module.num_layers):
                    layer_cell = o_module.layers[i].layer_fw.cell
                    w_ih, b_ih = layer_cell.igates.weight().dequantize(), layer_cell.igates.bias().dequantize()
                    w_hh, b_hh = layer_cell.hgates.weight().dequantize(), layer_cell.hgates.bias().dequantize()
                    new_state_dict.update({
                        f'{p_name}.weight_ih_l{i}': w_ih, f'{p_name}.weight_hh_l{i}': w_hh,
                        f'{p_name}.bias_ih_l{i}': b_ih, f'{p_name}.bias_hh_l{i}': b_hh
                    })

            elif not list(p_module.children()) and not isinstance(o_module, (nn.quantized.Linear, nn.quantized.Conv2d, nn.quantized.LSTM)):
                 if hasattr(o_module, 'state_dict') and o_module.state_dict():
                    print(f"  - Loading weights for float layer: '{p_name}'")
                    for param_name, param_value in o_module.state_dict().items():
                        new_state_dict[f"{p_name}.{param_name}"] = param_value

    placeholder_model.load_state_dict(new_state_dict, strict=False)

    # --- Part 2: Copy scale/zero_point to the HardcodedQuantizer modules ---
    print("\n[INFO] Copying scale/zp for accelerator boundaries...")
    
    # In the converted model, the scale/zp from the stub are absorbed by the *next* layer.
    # We access them from there.
    for i in range(len(placeholder_model.quant_attention)):
        # For the Attention block's input quantization
        source_attn_linear = original_model.attention_blocks[i].q_proj
        scale = source_attn_linear.scale
        zero_point = source_attn_linear.zero_point
        print(f"  - Updating quant_attention.{i} with scale={scale:.4f}, zp={zero_point}")
        placeholder_model.quant_attention[i].scale.copy_(torch.tensor(scale))
        placeholder_model.quant_attention[i].zero_point.copy_(torch.tensor(zero_point))
        
        # For the FFN block's input quantization
        source_ffn_linear = original_model.ffn_blocks[i].fc1
        scale = source_ffn_linear.scale
        zero_point = source_ffn_linear.zero_point
        print(f"  - Updating quant_ffn.{i} with scale={scale:.4f}, zp={zero_point}")
        placeholder_model.quant_ffn[i].scale.copy_(torch.tensor(scale))
        placeholder_model.quant_ffn[i].zero_point.copy_(torch.tensor(zero_point))

    print("\n✅ Transfer complete. Placeholder model is ready for export.")
    return placeholder_model
