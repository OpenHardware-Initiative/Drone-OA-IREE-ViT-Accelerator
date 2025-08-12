#
# FILE: quantize_ita_model_torchao.py
#
import torch
import torch.nn as nn
import copy
import os
import sys

# Core torchao imports for the quantization flow
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.export_utils import _move_exported_model_to_eval, _allow_exported_model_train_eval
from torch_mlir.fx import export_and_import
from torch_mlir.extras.fx_decomp_util import get_decomposition_table


# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import your model and the custom quantizer components
from models.ITA.ITA_model import ITALSTMNetVITFloat
from models.ITA.ITA_layers import ITASoftmax, OverlapPatchMerging, ITASelfAttentionFloat, ITAFeedForwardFloat  # Ensure this is used in the model
from ita_quantizer import ITAQuantizer
from ita_quantization_specs import get_ita_accelerator_qconfig, get_ita_softmax_qconfig, get_arm_symmetric_qconfig

def main():
    print("Step 1: Initializing float model and custom quantizer...")
    float_model = ITALSTMNetVITFloat().eval()
    
    example_inputs = ((
        torch.randn(1, 1, 60, 90), torch.randn(1, 1), torch.randn(1, 4), None,
    ),)

    # --- Part 2: Configure the Module-Aware Quantizer ---
    print("\nStep 2: Configuring the module-aware ITAQuantizer...")
    quantizer = ITAQuantizer()
    
    # Create the two different quantization configs
    arm_config = get_arm_symmetric_qconfig()
    accelerator_config = get_ita_accelerator_qconfig()
    softmax_config = get_ita_softmax_qconfig()

    # Assign configs to module types
    quantizer.set_config_for(OverlapPatchMerging, arm_config)
    quantizer.set_config_for(nn.LSTM, arm_config)
    quantizer.set_config_for(ITASelfAttentionFloat, accelerator_config)
    quantizer.set_config_for(ITAFeedForwardFloat, accelerator_config)
    quantizer.set_config_for(ITASoftmax, softmax_config)
    # The final nn.Linear and LayerNorms are not in custom modules,
    # so we need to give them a default config.
    quantizer.set_config_for(nn.Linear, arm_config)
    quantizer.set_config_for(nn.LayerNorm, arm_config)
    

    # --- Part 3: Quantization and Export ---
    print("\nStep 3: Capturing model graph...")
    exported_model = torch.export.export(
        float_model, args=copy.deepcopy(example_inputs)
    ).module()
    
    decomposed_program = exported_model.run_decompositions(get_decomposition_table())
    decomposed_model = decomposed_program.module()
    
    print("\nStep 4: Preparing model for quantization...")
    prepared_model = prepare_pt2e(exported_model, quantizer)

    print("\nStep 5: Calibrating the model...")
    with torch.no_grad():
        prepared_model(*example_inputs)
    
    print("\nStep 6: Converting the model...")
    quantized_program = convert_pt2e(prepared_model)
    
    print("\nStep 6.5: Sanitizing graph by converting Parameters to Buffers...")
    param_names = [name for name, _ in quantized_program.named_parameters()]
    for name in param_names:
        # getattr can't handle nested names with dots, so we navigate the module hierarchy
        parent_module = quantized_program
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        
        param_name = name_parts[-1]
        # Get the tensor data from the parameter
        param_data = getattr(parent_module, param_name).data
        
        # Delete the old parameter and register the plain tensor as a buffer
        delattr(parent_module, param_name)
        parent_module.register_buffer(param_name, param_data)
    
    # After this loop, quantized_program has no .parameters() left, only .buffers()
    # which are plain tensors. This satisfies the torch-mlir importer.

    # =========================================================================
    # Step 7: Lowering to MLIR
    # =========================================================================
    print("\nStep 7: Compiling the sanitized program to MLIR...")
    
    
    mlir_module = export_and_import(
        quantized_program,      # This now correctly maps to 'f'
        *example_inputs,        # This now correctly maps to '*args'
        #output_type="linalg-on-tensors",
        enable_ir_printing=True,
    )

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_output_path = os.path.join(output_dir, "ita_quantized.mlir")
    print(f"Saving MLIR to {mlir_output_path}...")
    with open(mlir_output_path, "w") as f:
        f.write(str(mlir_module))

    print("\n✅ Successfully lowered model to MLIR.")
    print("You can now compile the .mlir file with IREE:")
    print(f"iree-compile --iree-hal-target-backends=llvm-cpu {mlir_output_path} -o output/ita_quantized.vmfb")
    
    #print("\nStep 7: Exporting the quantized model to ONNX...")
    #output_dir = "output"
    #os.makedirs(output_dir, exist_ok=True)
    #onnx_path = os.path.join(output_dir, "ita_quantized_model.onnx")
#
    #torch.onnx.export(
    #    eval_quantized_program,
    #    args=copy.deepcopy(example_inputs),
    #    f=onnx_path,
    #    #dynamo=True,
    #    report=True,
    #    # Update input_names if your model signature changed
    #    input_names=['x_0', 'x_1', 'x_2', 'x_3'], # Corrected names
    #    output_names=['add_55', 'split_dim_28', 'split_dim_29'],
    #    opset_version=17
    #)
#
    #print(f"\n✅ Successfully exported quantized model to: {onnx_path}")

if __name__ == "__main__":
    main()
