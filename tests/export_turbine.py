import torch
print("\nInstalled PyTorch, version:", torch.__version__)

import iree.turbine.aot as aot

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.ITA.ITA_model import ITALSTMNetVITFloat

torch.manual_seed(0)

QUANTIZED_MODEL_PATH = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/quantized_model_final.pth" 
DEVICE = "cpu"

model = ITALSTMNetVITFloat()
model.eval()

model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#model.swap_modules_for_hardware()
model_prepared = torch.quantization.prepare(model, inplace=False)
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

print("Model structure successfully converted to quantized format.")
print(model_quantized)

# --- Step 3: Load the saved quantized weights and biases ---
print(f"\n--- Step 3: Loading weights from {QUANTIZED_MODEL_PATH} ---")

# Load the state dictionary from the file
state_dict = torch.load(QUANTIZED_MODEL_PATH, map_location=DEVICE)

model_quantized.load_state_dict(state_dict)

# --- FIX: Create Example Inputs for Tracing ---
print("\n--- Creating example inputs for tracing ---")
# Your model's forward pass logic dictates these shapes:
# 1. Image: [batch, channels, height, width] -> [1, 1, 60, 90]
img_input = torch.randn(1, 1, 60, 90)

# 2. Additional Data (e.g., desvel): [batch, features] -> [1, 1]
add_input = torch.randn(1, 1)

# 3. Quaternion Data: [batch, features] -> [1, 4]
quat_input = torch.randn(1, 4)

# 4. LSTM Hidden State: A tuple of (h_0, c_0).
#    Shape is (num_layers, batch_size, hidden_size)
#    Your LSTM has num_layers=3, hidden_size=128
h0 = torch.randn(3, 1, 128)
c0 = torch.randn(3, 1, 128)
hidden_state = (h0, c0)

# Combine into a list, as expected by your forward(self, X) method
example_args = [img_input, add_input, quat_input, hidden_state]


# --- Export the model using the example inputs ---
print("\n--- Exporting model to MLIR ---")
export_model = aot.export(model_quantized, args=(example_args,)) 

mlir_file_path = "/tmp/linear_module_pytorch.mlir"

export_model.print_readable()
export_model.save_mlir(mlir_file_path)


#class CompiledModel(aot.CompiledModule):
#    params = aot.export_parameters(model, mutable=True)
#    compute = aot.jittable(model.forward)
#    
#    def main(self, x=aot.AbstractTensor(4)):
#        return self.compute(x)
#
#    def get_weight(self):
#        return self.params["weight"]
#
#    def set_weight(self, weight=aot.abstractify(example_weight)):
#        self.params["weight"] = weight
#
#    def get_bias(self):
#        return self.params["bias"]
#
#    def set_bias(self, bias=aot.abstractify(example_bias)):
#        self.params["bias"] = bias

