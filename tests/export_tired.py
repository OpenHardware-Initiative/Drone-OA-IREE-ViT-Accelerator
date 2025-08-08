import torch
import torch.ao.quantization.quantize_pt2e as quantize_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.qconfig import get_default_pt2e_qconfig
import torch_mlir
import os
import sys

# --- Setup and Imports ---
print("\nInstalled PyTorch, version:", torch.__version__)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.ITA.ITA_model import ITALSTMNetVITFloat
# You will need your dataloader to calibrate the model
from third_party.vitfly_FPGA.training.dataloading import dataloader, preload

# --- Configuration ---
# ❗️ IMPORTANT: Use your trained FLOAT model path here
FLOAT_MODEL_PATH = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/model_000004.pth"
DATASET_DIR = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/data"
MLIR_PATH = "ita_quantized_pt2e.mlir" # Output path for your MLIR file
DEVICE = "cpu"

# --- Step 1: Load the TRAINED FLOAT model ---
print("\n--- Step 1: Loading trained float model ---")
model_float = ITALSTMNetVITFloat().to(DEVICE)
model_float.load_state_dict(torch.load(FLOAT_MODEL_PATH, map_location=DEVICE))
model_float.eval()

# --- Step 2: Create Example Inputs for Export Tracing ---
print("\n--- Step 2: Creating example inputs for export ---")
img_input = torch.randn(1, 1, 60, 90)
add_input = torch.randn(1, 1)
quat_input = torch.randn(1, 4)
h0 = torch.randn(3, 1, 128)
c0 = torch.randn(3, 1, 128)
hidden_state = (h0, c0)
# Note the extra comma to make it a tuple of one argument (a list)
example_args = ([img_input, add_input, quat_input, hidden_state],)

# --- Step 3: Prepare the model for PT2E Quantization ---
print("\n--- Step 3: Preparing model for PT2E quantization ---")
quantizer = X86InductorQuantizer()
quantizer.set_global(get_default_pt2e_qconfig())
# 'prepare' inserts observers into the model graph
prepared_model = quantize_pt2e.prepare(model_float, quantizer, example_args)
print("Model prepared for quantization.")

# --- Step 4: Calibrate the model with real data ---
print("\n--- Step 4: Calibrating model ---")
train_data, _, _, _ = dataloader(DATASET_DIR, val_split=0.99, short=5, seed=42)
train_meta, train_ims, _, train_desvel, train_currquat, _ = train_data
train_ims, train_desvel, train_currquat = preload((train_ims, train_desvel, train_currquat), DEVICE)

with torch.no_grad():
    print("Running calibration...")
    for i in range(min(10, len(train_ims))): # Calibrate on ~10 samples
        img = train_ims[i].unsqueeze(0).unsqueeze(0)
        desvel = train_desvel[i].view(1, 1)
        quat = train_currquat[i].unsqueeze(0)
        # Your model's forward pass handles the case where the hidden state might not
        # be in the dataset by using a default. We replicate that here.
        calibration_input = [img, desvel, quat, hidden_state]
        prepared_model(calibration_input) # Run data through the prepared model
print("Calibration complete.")

# --- Step 5: Convert the model to its final quantized graph form ---
print("\n--- Step 5: Converting to final quantized model ---")
# 'convert' uses the observer stats to create the final quantized graph
quantized_model_graph = quantize_pt2e.convert(prepared_model)
print("Model converted to a quantized graph.")

# --- Step 6: Export the Quantized Graph to an MLIR File ---
print(f"\n--- Step 6: Exporting to Torch-MLIR at {MLIR_PATH} ---")
mlir_module = torch_mlir.compile(
    quantized_model_graph,
    example_args,
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=False # PT2E models are already graphs, no need to trace again
)

with open(MLIR_PATH, "w") as f:
    f.write(str(mlir_module))

print(f"\n✅ MLIR file successfully saved to {MLIR_PATH}")