# quantize_ptq.py

import torch
import torch.quantization



import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.ITA.ITA_model import ITALSTMNetVITFloat
# You will need your dataloader to calibrate the model
from third_party.vitfly_FPGA.training.dataloading import dataloader, preload 

# --- Configuration ---
FLOAT_MODEL_PATH = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/model_000004.pth" # 👈 UPDATE THIS
QUANTIZED_MODEL_PATH = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/quantized_model_final.pth" # 👈 UPDATE THIS
DATASET_DIR = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/data" # 👈 UPDATE THIS for calibration
DEVICE = "cpu" # Quantization is typically done on the CPU

# 1. Load the trained float model
print("Loading trained float model...")
model_float = ITALSTMNetVITFloat()
model_float.load_state_dict(torch.load(FLOAT_MODEL_PATH, map_location=DEVICE))
model_float.eval()

# 2. Swap standard modules with your custom hardware-compliant ones
#print("Swapping float modules for hardware-specific integer modules...")
#model_float.swap_modules_for_hardware()

# 3. Configure the quantization scheme
# This matches your desired hardware constraints: symmetric int8
# NOTE: We use Per-Channel quantization for weights (Conv, Linear) for better accuracy,
# and Per-Tensor for activations, as is standard practice.
qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False
    )
)

model_float.qconfig = qconfig

# 4. Prepare the model for PTQ
print("Preparing model for PTQ...")
model_prepared = torch.quantization.prepare(model_float, inplace=False)

# 5. Calibrate the model with real data
# Calibration feeds data through the model to determine the activation ranges (scale/zero-point).
print("Calibrating model...")
# --- Load a small amount of calibration data ---
train_data, _, _, _ = dataloader(DATASET_DIR, val_split=0.99, short=5, seed=42)
# Unpack all the data arrays from the dataloader's output
train_meta, train_ims, _, train_desvel, train_currquat, _ = train_data

# --- Preload ONLY the necessary numpy arrays into tensors ---
# We avoid passing None to the function
preloaded_tensors = preload((train_ims, train_desvel, train_currquat), DEVICE)
# Unpack the results
train_ims, train_desvel, train_currquat = preloaded_tensors


# --- Run a few batches through the model ---
with torch.no_grad():
    for i in range(min(10, len(train_ims))): # Calibrate on ~10 samples
        img = train_ims[i].unsqueeze(0).unsqueeze(0)
        desvel = train_desvel[i].view(1, 1)
        quat = train_currquat[i].unsqueeze(0)
        model_prepared([img, desvel, quat, None])


print("Calibration complete.")

# 6. Convert the model to a fully quantized version
print("Converting to fully quantized model...")
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# 7. Save the final quantized model
print(f"Saving quantized model to {QUANTIZED_MODEL_PATH}")
torch.save(model_quantized.state_dict(), QUANTIZED_MODEL_PATH)

print("\nPTQ process finished successfully!")
print("Final model is ready for MLIR export.")