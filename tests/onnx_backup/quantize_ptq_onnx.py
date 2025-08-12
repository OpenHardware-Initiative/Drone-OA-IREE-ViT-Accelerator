# quantize_onnx.py

import torch
import torch.nn as nn
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

import os
import sys
import numpy as np

# --- 1. Setup & Configuration ---

# Add project root to path to allow importing model/data loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.ITA.ITA_model import ITALSTMNetVITFloat
from third_party.vitfly_FPGA.training.dataloading import dataloader, preload

# --- Configuration ---
# 👉 UPDATE THESE PATHS
FLOAT_MODEL_PATH = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/model_000004.pth"
DATASET_DIR = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/data"
OUTPUT_DIR = "/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig" # Directory to save ONNX and MLIR files

# Derived paths
FLOAT_ONNX_PATH = os.path.join(OUTPUT_DIR, "model_float.onnx")
QUANTIZED_ONNX_PATH = os.path.join(OUTPUT_DIR, "model_quantized.onnx")
MLIR_PATH = os.path.join(OUTPUT_DIR, "model_quantized.mlir")

DEVICE = "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_nodes_between(model_path, start_node_names, end_node_names):
    """
    Finds all nodes between a set of start and end nodes in an ONNX graph.

    Args:
        model_path (str): Path to the ONNX model.
        start_node_names (list[str]): Names of the nodes that are right BEFORE the subgraph.
        end_node_names (list[str]): Names of the nodes that are right AFTER the subgraph.

    Returns:
        list[str]: A list of names of the nodes in the subgraph.
    """
    model = onnx.load(model_path)
    # Create lookups for producers and consumers of each tensor
    node_outputs = {}
    consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in consumers:
                consumers[input_name] = []
            consumers[input_name].append(node)
        for output_name in node.output:
            node_outputs[output_name] = node

    subgraph_nodes = set()
    
    # Get the output tensors of the start nodes
    start_tensors = []
    for node in model.graph.node:
        if node.name in start_node_names:
            start_tensors.extend(node.output)

    # Use a queue for a Breadth-First Search (BFS)
    queue = []
    for tensor in start_tensors:
        if tensor in consumers:
            queue.extend(consumers[tensor])
            
    visited = set()
    while queue:
        current_node = queue.pop(0)
        if current_node.name in visited or current_node.name in end_node_names:
            continue
        
        visited.add(current_node.name)
        subgraph_nodes.add(current_node.name)
        
        # Add the next nodes in the graph to the queue
        for output_tensor in current_node.output:
            if output_tensor in consumers:
                for consumer_node in consumers[output_tensor]:
                    queue.append(consumer_node)
                    
    return list(subgraph_nodes)


# --- 2. Model Preparation for ONNX Export ---

class ITALSTMNetVITFloatONNX(ITALSTMNetVITFloat):
    """
    A wrapper around the original model with a modified forward pass,
    making it compatible with torch.onnx.export. It accepts individual
    tensors instead of a list.
    """
    def forward(self, img_data, additional_data, quat_data, h_in, c_in):
        # The 'refine_inputs' logic from the original model is now assumed
        # to be handled by the dataloader. The ONNX graph should be static.
        # We explicitly pass LSTM states for a clear graph definition.
        
        x, _, _ = self.tokenizer(img_data)
        x = x.view(1, 128, self.E)
        
        for i in range(len(self.attention_blocks)):
            x = self.norm1_layers[i](x + self.attention_blocks[i](x))
            x = self.norm2_layers[i](x + self.ffn_blocks[i](x))
        
        x = x.flatten(1)
        out = self.decoder(x)
        
        # Concatenate features
        out_cat = torch.cat([out, additional_data / 10.0, quat_data], dim=1).unsqueeze(0)
        
        # Use explicit hidden states for LSTM
        hidden_state = (h_in, c_in)
        out_lstm, (h_out, c_out) = self.lstm(out_cat, hidden_state)
        
        out_final = self.nn_fc2(out_lstm.squeeze(0))
        
        # Return final output and the updated LSTM states
        return out_final, h_out, c_out


print("✅ Step 1: Loading trained float model...")
model_float = ITALSTMNetVITFloatONNX() # Use the ONNX-compatible version
model_float.load_state_dict(torch.load(FLOAT_MODEL_PATH, map_location=DEVICE))
model_float.eval()


# --- 3. Export to Floating-Point ONNX ---

print("\n✅ Step 2: Exporting to floating-point ONNX...")

# Create dummy inputs that match the expected shapes for the ONNX graph.
# Batch size is 1.
dummy_img = torch.randn(1, 1, 60, 90, device=DEVICE)
# 🚨 FIX IS HERE: The `additional_data` (desvel) feature dimension should be 1, not 3.
dummy_desvel = torch.randn(1, 1, device=DEVICE) 
dummy_quat = torch.randn(1, 4, device=DEVICE)
# LSTM states: (num_layers, batch_size, hidden_size)
dummy_h_in = torch.randn(3, 1, 128, device=DEVICE)
dummy_c_in = torch.randn(3, 1, 128, device=DEVICE)
dummy_inputs = (dummy_img, dummy_desvel, dummy_quat, dummy_h_in, dummy_c_in)

# Define input and output names for the ONNX graph
input_names = ["img_data", "additional_data", "quat_data", "h_in", "c_in"]
output_names = ["final_output", "h_out", "c_out"]

torch.onnx.export(
    model_float,
    dummy_inputs,
    FLOAT_ONNX_PATH,
    input_names=input_names,
    output_names=output_names,
    #dynamo=True,
    opset_version=17, # A modern opset is recommended
    verbose=True,
    dynamic_axes={ # If batch size can vary
        'img_data': {0: 'batch_size'},
        'additional_data': {0: 'batch_size'},
        'quat_data': {0: 'batch_size'},
        'h_in': {1: 'batch_size'},
        'c_in': {1: 'batch_size'},
        'final_output': {0: 'batch_size'},
        'h_out': {1: 'batch_size'},
        'c_out': {1: 'batch_size'}
    }
)
print(f"Float ONNX model saved to {FLOAT_ONNX_PATH}")


# --- 4. Calibrate and Quantize with ONNX Runtime ---

print("\n✅ Step 3: Calibrating and quantizing with ONNX Runtime...")

class CalibrationDataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, dataset_dir, device):
        print("Initializing calibration data reader...")
        # Load a small, representative subset of the data
        train_data, _, _, _ = dataloader(dataset_dir, val_split=0.99, short=20, seed=42)
        
        # 🚨 FIX IS HERE: Unpack all 6 elements from train_data tuple to match dataloading.py
        # We only need train_ims, train_desvel, and train_currquat for calibration.
        _train_meta, train_ims, _train_trajlength, train_desvel, train_currquat, _train_currctbr = train_data

        # Preload the necessary data to tensors and then convert to numpy
        preloaded = preload((train_ims, train_desvel, train_currquat), device)
        self.img_data = preloaded[0].numpy()
        self.desvel_data = preloaded[1].numpy()
        self.quat_data = preloaded[2].numpy()

        # Create initial zero-state numpy arrays for the LSTM
        self.h_in = np.zeros((3, 1, 128), dtype=np.float32)
        self.c_in = np.zeros((3, 1, 128), dtype=np.float32)

        # Create an iterator that maps input names to numpy data
        self.data_iterator = iter([
            {
                "img_data": img.reshape(1, 1, *img.shape),
                "additional_data": desvel.reshape(1, 1), # Ensure shape is (1, 1)
                "quat_data": np.expand_dims(quat, 0),
                "h_in": self.h_in,
                "c_in": self.c_in
            }
            for img, desvel, quat in zip(self.img_data, self.desvel_data, self.quat_data)
        ])
        print(f"Calibration data loaded. Number of samples: {len(self.img_data)}")

    def get_next(self):
        return next(self.data_iterator, None)


# Instantiate the reader
calibration_data_reader = CalibrationDataReader(DATASET_DIR, DEVICE)

print("\n✅ Step 3: Finding Softmax output tensor name...")
softmax_output_tensor_name = None
onnx_model = onnx.load(FLOAT_ONNX_PATH)
for node in onnx_model.graph.node:
    if node.op_type == 'Softmax':
        # The output tensor is the first (and only) output of the Softmax node
        softmax_output_tensor_name = node.output[0] 
        print(f"   🎯 Found Softmax node '{node.name}', its output tensor is: '{softmax_output_tensor_name}'")
        break

if not softmax_output_tensor_name:
    print("   ❌ Error: Could not find a Softmax node in the ONNX graph. Exiting.")
    #sys.exit(1)
    
forced_scale = 1.0 / 128.0
forced_zero_point = 127

# Define the tensor-specific override
tensor_overrides = {
    softmax_output_tensor_name: [
        {
            'quant_type': QuantType.QUInt8,
            'scale': np.float32(forced_scale),
            'zero_point': np.uint8(forced_zero_point),
        }
    ]
}
# Perform static quantization
quantize_static(
    model_input=FLOAT_ONNX_PATH,
    model_output=QUANTIZED_ONNX_PATH,
    calibration_data_reader=calibration_data_reader,
    quant_format=QuantFormat.QDQ,  # Standard format
    activation_type=QuantType.QUInt8,   # Symmetric INT8 for activations
    weight_type=QuantType.QInt8,       # Symmetric INT8 for weights
    #op_types_to_quantize_per_channel=['MatMul', 'Conv'], # Per-channel for Conv/Linear
    per_channel=False, # Enable per-channel quantization for weights
    extra_options={
        'ActivationSymmetric': True, # Enforce symmetric quantization
        'WeightSymmetric': True,      # Enforce symmetric quantization
        #'TensorQuantOverrides': tensor_overrides,
        #'ForceQuantizeNoInputCheck': True,
    }
)
print(f"Quantized ONNX model saved to {QUANTIZED_ONNX_PATH}")


# --- 5. Export to MLIR ---

print("\n✅ Step 4: Exporting to MLIR...")
print("To convert the quantized ONNX model to MLIR, run the following command in your terminal:")
print("-" * 70)
# Use IREE's (Intermediate Representation and Execution Environment) tool
# You may need to add flags to target your specific hardware accelerator,
# for example: --iree-hal-target-backends=...
mlir_command = f"iree-import-onnx {QUANTIZED_ONNX_PATH} --opset-version 17 -o {MLIR_PATH} "
print(f"👉 \033[1m{mlir_command}\033[0m")
print("-" * 70)
print("\nPTQ process with ONNX and MLIR export instructions are complete! 🎉")