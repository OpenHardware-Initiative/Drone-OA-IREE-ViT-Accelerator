import onnx
from onnx import numpy_helper
import numpy as np

# --- Configuration ---
# The full path to your ONNX model file
ONNX_FILE_PATH = '/home/coppholl/Projects/Drone-OA-IREE-ViT-Accelerator/training/logs/d08_07_t15_06_float_trainig/model_quantized.onnx'

# The exact name of the tensor you want to extract
TENSOR_NAME = 'attention_blocks.0.k_proj.bias_quantized'


# --- Main Script ---
def extract_tensor(model_path, tensor_name):
    """
    Loads an ONNX model and extracts a specific tensor from its initializers.
    """
    try:
        # 1. Load the ONNX model from the file
        model = onnx.load(model_path)
        print(f"✅ Successfully loaded model: {model_path}")

        # 2. Search for the tensor in the model's graph initializers
        target_tensor_proto = None
        for initializer in model.graph.initializer:
            if initializer.name == tensor_name:
                target_tensor_proto = initializer
                break
        
        # 3. Check if the tensor was found and convert it
        if target_tensor_proto:
            # Convert the ONNX TensorProto to a NumPy array
            numpy_array = numpy_helper.to_array(target_tensor_proto)
            
            print("\n--- Tensor Found! ---")
            print(f"Name: {tensor_name}")
            print(f"Data Type: {numpy_array.dtype}")
            print(f"Shape: {numpy_array.shape}")
            print("\nValues:")
            print(numpy_array)
            print("---------------------\n")
            return numpy_array
        else:
            print(f"❌ Error: Tensor '{tensor_name}' not found in the model's initializers.")
            return None

    except FileNotFoundError:
        print(f"❌ Error: The file was not found at the specified path: {model_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Run the extraction function
extracted_bias = extract_tensor(ONNX_FILE_PATH, TENSOR_NAME)