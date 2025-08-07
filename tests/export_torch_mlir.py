import torch
import torch.nn as nn
import iree.turbine.aot as aot
import os

# Import the model class you want to export.
# This must have an architecture compatible with your saved weights.
from ita_model_export import ITALSTMNetVIT_Export

# --- 1. Model Preparation and Weight Loading ---

# Path to the weights file saved by your QATTrainer's `finalize` method.
PATH_TO_WEIGHTS = "path/to/your/workspace/model_quantized_final.pth"

# Instantiate the model architecture you intend to export.
# This is the "blueprint".
export_model = ITALSTMNetVIT_Export()

# Load the trained weights from your .pth file into the model instance.
# This "furnishes" the house.
print(f"🛠️  Loading trained weights from: {PATH_TO_WEIGHTS}")
# Use map_location to ensure the model loads to the CPU, which is ideal for export.
state_dict = torch.load(PATH_TO_WEIGHTS, map_location='cpu')

# The .load_state_dict() method places the loaded weights into the model.
# We use strict=False because the saved state_dict is from a "converted" QAT model,
# while our export_model still has standard nn.Linear layers. This allows it
to
# load all matching parameters (like 'attention_blocks.0.q_proj.weight')
# while ignoring keys that may no longer match perfectly.
export_model.load_state_dict(state_dict, strict=False)

# Set the now-populated model to evaluation mode.
export_model.eval()


# --- 2. Create the Export Wrapper ---

# As before, we wrap the model to ensure we export the
# `forward_with_intermediates` method.
class IntermediateWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model.forward_with_intermediates(X)

wrapped_model = IntermediateWrapper(export_model)


# --- 3. Create Example Inputs for Tracing ---

batch_size = 1
img_data = torch.randn(batch_size, 1, 60, 90, dtype=torch.float32)
example_args = [[img_data]]


# --- 4. Export to MLIR ---

print("🚀 Starting model export to MLIR...")
exported_program = aot.export(wrapped_model, args=example_args)


# --- 5. Save the MLIR Code ---

output_mlir_path = "ita_model_final_trained.mlir"
exported_program.save_mlir(output_mlir_path)

print(f"✅ Trained model successfully exported to MLIR at: '{os.path.abspath(output_mlir_path)}'")