import torch
import torch.nn as nn
import iree.turbine.aot as aot
import os
import sys

from torch.quantization import QConfig, FusedMovingAvgObsFakeQuantize, prepare_qat, convert


# Import the model class you want to export.
# This must have an architecture compatible with your saved weights.
# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from models.ITA.export.ITA_model_export import ITALSTMNetVIT_Export
from models.ITA.ITA_model import ITALSTMNetVIT

# --- 1. Model Preparation and Weight Loading ---

# Path to the weights file saved by your QATTrainer's `finalize` method.
PATH_TO_WEIGHTS = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t13_38_qat_rom_scratch/model_quantized_final.pth"

# Instantiate the model architecture you intend to export.
# This is the "blueprint".
export_model = ITALSTMNetVIT()

torch.backends.quantized.engine = 'qnnpack'

# Define the quantization configuration. This MUST match the one used during training.
# CRITICAL: Activations use qint8 (signed) to match hardware's unified data path.
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


export_model.qconfig = ita_symmetric_qconfig
export_model.tokenizer.qconfig = None # The tokenizer runs on the CPU and is not quantized.
for block in export_model.norm1_layers:
    block.qconfig = None
for block in export_model.norm2_layers:
    block.qconfig = None

# Convert the model to its final integer-only representation.
torch.quantization.prepare_qat(export_model, inplace=True)
torch.quantization.convert(export_model, inplace=True)
export_model.eval()

# Load the trained weights from your .pth file into the model instance.
# This "furnishes" the house.
print(f"🛠️  Loading trained weights from: {PATH_TO_WEIGHTS}")
# Use map_location to ensure the model loads to the CPU, which is ideal for export.
state_dict = torch.load(PATH_TO_WEIGHTS, map_location='cpu')

# The .load_state_dict() method places the loaded weights into the model.
# We use strict=False because the saved state_dict is from a "converted" QAT model,
# while our export_model still has standard nn.Linear layers. This allows it
# load all matching parameters (like 'attention_blocks.0.q_proj.weight')
# while ignoring keys that may no longer match perfectly.
export_model.load_state_dict(state_dict, strict=False)

# Set the now-populated model to evaluation mode.
export_model.eval()





# --- 3. Create Example Inputs for Tracing ---

batch_size = 1
img_data = torch.randn(batch_size, 1, 60, 90, dtype=torch.float32)

# Your forward pass requires `X` to have at least 3 elements for unpacking.
# We must provide them. `X[1]` is `additional_data`. Based on your model's
# LSTM input size (517) vs. the decoder output (512) and quat_data (4),
# this tensor should have a feature size of 1.
additional_data = torch.randn(batch_size, 1, dtype=torch.float32)

# We can pass `None` for quat_data (at index 2) and the optional hidden_state
# (at index 3) and let `refine_inputs` and the forward pass handle them.
# The crucial part is that the list has enough elements to avoid the IndexError.
model_input_list = [img_data, additional_data, None, None]

# The collection of arguments for the export function must be a TUPLE.
example_args = (model_input_list,)


# --- 4. Export to MLIR ---

print("🚀 Starting model export to MLIR...")
exported_program = aot.export(export_model, args=example_args)


# --- 5. Save the MLIR Code ---

output_mlir_path = "ita_model_final_trained.mlir"
exported_program.save_mlir(output_mlir_path)

print(f"✅ Trained model successfully exported to MLIR at: '{os.path.abspath(output_mlir_path)}'")