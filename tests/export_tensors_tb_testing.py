import torch
import numpy as np
import argparse
import os
import sys
from PIL import Image
from torchvision import transforms
from torch.quantization import get_default_qat_qconfig
from torch.quantization import QConfig, FusedMovingAvgObsFakeQuantize

# --- Ensure correct paths for imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# --- Project-Specific Imports ---
from models.testing.export.ITA_model_export import ITALSTMNetVIT_Export
from models.testing.ITA_layers import ITASoftmax
from third_party.ITA_FPGA.PyITA.util import (
    write_matrix, to_hex, pack_hex_24b, pack_array_8b_to_word, pack_8b_to_word,
    write_vector_mem_hex, write_matrix_mem_hex, split_matrix, generate_matrix_mem,
    write_matrix_mem
)

# ==============================================================================
# HELPER: Scale to Multiplier/Shift Conversion
# ==============================================================================

def calculate_multiplier_shift(effective_scale, bit_width=31):
    """
    Calculates the integer multiplier and bit-shift required to approximate
    a floating-point multiplication.
    Hardware operation: (input * mult) >> shift
    """
    if effective_scale == 0:
        return 0, 0
    # Represents the float scale as a fixed-point number: multiplier / 2^bit_width
    multiplier = int(round(effective_scale * (2**bit_width)))
    return multiplier, bit_width

# ==============================================================================
# CLASS 1: HARDWARE SIMULATION ("GOLDEN MODEL")
# Encapsulates the logic for simulating the hardware's exact integer arithmetic.
# This keeps the simulation logic separate from the data extraction and file writing.
# ==============================================================================

class HardwareGoldenModel:
    """
    Simulates the ITA hardware operations step-by-step using NumPy.
    This model uses the weights, biases, and requantization parameters
    extracted from the trained PyTorch model to generate bit-accurate results.
    """
    def __init__(self, model_dims, hw_params):
        self.p = model_dims
        self.hw_params = hw_params
        self.tensors = {}
        # Use the PyTorch implementation of ITASoftmax for bit-accuracy with the reference
        self.softmax_sim = ITASoftmax()
        print("✨ Golden Model Initialized.")

    def _np_requantize(self, x, mult, shift, add=0):
        x = x.astype(np.float64) * mult
        x = np.floor(x / (2**shift) + 0.5) + add
        return np.clip(x, -128, 127).astype(np.int8)

    def _np_i_gelu(self, x_in):
        params = self.hw_params['gelu']
        q_1, q_b, q_c = params['q_1'], params['q_b'], params['q_c']
        mult, shift, add = params['rqs_mul'], params['rqs_shift'], params['rqs_add']
        
        # Inner functions defining the polynomial approximation of GELU
        def _i_poly(q, b, c):
            d = q.astype(np.int16) + b
            return (d * d + c).astype(np.int32)
        def _i_erf(q, b, c):
            q_sgn = np.sign(q)
            q_abs = np.abs(q)
            q_clipped = np.clip(q_abs, 0, -b)
            return q_sgn * _i_poly(q_clipped, b, c)
        def _i_gelu(q, one, b, c):
            q_clipped = np.clip(q, -127, 127)
            q_erf = _i_erf(q_clipped, b, c)
            return q_clipped * (q_erf + one)

        q_out = _i_gelu(x_in, q_1, q_b, q_c)
        # The output of the GELU approximation is also requantized
        return self._np_requantize(q_out, mult, shift, add)

    def load_parameters(self, block_tensors):
        """Loads the extracted weights, biases, and inputs for the current block."""
        self.tensors.update(block_tensors)
        print("  ✅ Loaded weights, biases, and inputs into golden model.")

    def run_simulation(self):
        """Executes the full hardware simulation for one transformer block."""
        print("  ✨ Recomputing all intermediates with hardware-like arithmetic...")
        S = self.p['S']

        # Attention Block Simulation
        self.tensors['Qp'] = np.matmul(self.tensors['Q_in'], self.tensors['Wq']) + np.tile(self.tensors['Bq'], [1, S, 1])
        self.tensors['Qp_requant'] = self._np_requantize(self.tensors['Qp'][0], **self.hw_params['q_proj'])[np.newaxis, :, :]

        self.tensors['Kp'] = np.matmul(self.tensors['K_in'], self.tensors['Wk']) + np.tile(self.tensors['Bk'], [1, S, 1])
        self.tensors['Kp_requant'] = self._np_requantize(self.tensors['Kp'][0], **self.hw_params['k_proj'])[np.newaxis, :, :]

        self.tensors['Vp'] = np.matmul(self.tensors['V_in'], self.tensors['Wv']) + np.tile(self.tensors['Bv'], [1, S, 1])
        self.tensors['Vp_requant'] = self._np_requantize(self.tensors['Vp'][0], **self.hw_params['v_proj'])[np.newaxis, :, :]

        A_matmul = np.matmul(self.tensors['Qp_requant'].astype(np.int32), self.tensors['Kp_requant'].transpose(0, 2, 1).astype(np.int32))
        self.tensors['A_requant'] = self._np_requantize(A_matmul[0], **self.hw_params['qk_matmul'])[np.newaxis, :, :]
        
        # Use torch ITASoftmax on numpy array for bit-accuracy
        self.tensors['A_partial_softmax'] = self.softmax_sim(torch.from_numpy(self.tensors['A_requant'])).numpy()

        O_soft_matmul = np.matmul(self.tensors['A_partial_softmax'].astype(np.int32), self.tensors['Vp_requant'].astype(np.int32))
        self.tensors['O_soft_requant'] = self._np_requantize(O_soft_matmul[0], **self.hw_params['av_matmul'])[np.newaxis, :, :]

        Out_soft_matmul = np.matmul(self.tensors['O_soft_requant'], self.tensors['Wo']) + np.tile(self.tensors['Bo'], [1, S, 1])
        self.tensors['Out_soft_requant'] = self._np_requantize(Out_soft_matmul[0], **self.hw_params['out_proj'])[np.newaxis, :, :]

        # FFN Block Simulation
        self.tensors['FFp'] = np.matmul(self.tensors['FF_in'], self.tensors['Wff']) + np.tile(self.tensors['Bff'], [1, S, 1])
        self.tensors['FFp_requant'] = self._np_requantize(self.tensors['FFp'][0], **self.hw_params['ffn1'])[np.newaxis, :, :]

        self.tensors['gelu'] = self._np_i_gelu(self.tensors['FFp_requant'][0])[np.newaxis, :, :]

        self.tensors['FF2p'] = np.matmul(self.tensors['gelu'], self.tensors['Wff2']) + np.tile(self.tensors['Bff2'], [1, S, 1])
        self.tensors['FF2p_requant'] = self._np_requantize(self.tensors['FF2p'][0], **self.hw_params['ffn2'])[np.newaxis, :, :]
        
        print("  ✅ Golden model simulation complete.")
        return self.tensors

# ==============================================================================
# CLASS 2: FILE WRITER
# Encapsulates all file I/O logic, ensuring formats match the Verilog testbenches.
# ==============================================================================
class FileWriter:
    """
    Handles all file generation for hardware verification, ensuring formats
    match the reference ITA.py script exactly.
    """
    def __init__(self, tensors, hw_params, model_dims, base_path):
        """
        Initializes the FileWriter.

        Args:
            tensors (dict): Dictionary of all simulated tensors (inputs, weights, intermediates).
            hw_params (dict): Dictionary of hardware parameters (mult, shift, add for each step).
            model_dims (dict): Dictionary of model dimensions (S, E, P, F, H).
            base_path (str): The root directory for the output files for a specific block.
        """
        self.t = tensors
        self.hw = hw_params
        self.p = model_dims
        self.paths = {
            "base": base_path,
            "hwpe": os.path.join(base_path, "hwpe"),
            "mempool": os.path.join(base_path, "mempool"),
            "standalone": os.path.join(base_path, "standalone"),
            "snitch": os.path.join(base_path, "snitch-cluster")
        }
        print(f"--- 💾 Initialized FileWriter for path: ./{base_path} ---")

    def create_directories(self):
        """Creates all necessary subdirectories for the output files."""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        print("  ✅ Created all output directories.")

    # --------------------------------------------------------------------
    # Private Helper Methods for Writing
    # --------------------------------------------------------------------

    def _write_flat_row(self, data, filename):
        """Writes a list or 1D array as a single space-separated row."""
        path = os.path.join(self.paths['base'], filename)
        np.savetxt(path, np.array([data]), fmt='%d', delimiter=' ')

    def _write_standalone_tensor(self, tensor, name):
        """Writes a tensor to the standalone folder, one value per line."""
        path = os.path.join(self.paths['standalone'], f"{name}.txt")
        # Standalone files expect a flattened 2D layout.
        if tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        with open(path, "w") as f:
            np.savetxt(f, tensor, fmt='%d')

    def _write_hwpe_tensor(self, tensor_np, filename):
        """Writes a tensor to the hwpe folder, packed into 32-bit hex."""
        filepath = os.path.join(self.paths['hwpe'], f'{filename}.txt')
        if os.path.exists(filepath):
             os.remove(filepath)
        
        # Squeeze to 2D, tile into 64x64 blocks, and pack to hex
        tiles = split_matrix(tensor_np.squeeze(0), block_shape=(64, 64))
        packed_hex = pack_array_8b_to_word(tiles, hex_string=False)
        write_matrix_mem_hex(packed_hex, filename, self.paths['hwpe'])

    # --------------------------------------------------------------------
    # Public Methods for Writing Different File Categories
    # --------------------------------------------------------------------

    def write_global_params(self):
        """Writes the top-level parameter files (RQS, GELU)."""
        # GELU Params
        self._write_flat_row([self.hw['gelu']['q_b']], "GELU_B.txt")
        self._write_flat_row([self.hw['gelu']['q_c']], "GELU_C.txt")
        self._write_flat_row([self.hw['gelu']['q_1']], "GELU_ONE.txt")
        self._write_flat_row([self.hw['gelu']['rqs_add']], "activation_requant_add.txt")
        self._write_flat_row([self.hw['gelu']['rqs_mul']], "activation_requant_mult.txt")
        self._write_flat_row([self.hw['gelu']['rqs_shift']], "activation_requant_shift.txt")

        # Requantization Params for Attention (7 steps: Q, K, V, QK, AV, OW, SumOW)
        # Note: FFN params are handled separately by the hardware controller logic
        rqs_attn_mul = [
            self.hw['q_proj']['mult'], self.hw['k_proj']['mult'], self.hw['v_proj']['mult'],
            self.hw['qk_matmul']['mult'], self.hw['av_matmul']['mult'], self.hw['out_proj']['mult'], 0
        ]
        rqs_attn_shift = [
            self.hw['q_proj']['shift'], self.hw['k_proj']['shift'], self.hw['v_proj']['shift'],
            self.hw['qk_matmul']['shift'], self.hw['av_matmul']['shift'], self.hw['out_proj']['shift'], 0
        ]
        rqs_attn_add = [
            self.hw['q_proj']['add'], self.hw['k_proj']['add'], self.hw['v_proj']['add'],
            self.hw['qk_matmul']['add'], self.hw['av_matmul']['add'], self.hw['out_proj']['add'], 0
        ]
        self._write_flat_row(rqs_attn_mul, "RQS_ATTN_MUL.txt")
        self._write_flat_row(rqs_attn_shift, "RQS_ATTN_SHIFT.txt")
        self._write_flat_row(rqs_attn_add, "RQS_ATTN_ADD.txt")
        
        # Requantization Params for FFN (2 steps: F1, F2)
        rqs_ffn_mul = [self.hw['ffn1']['mult'], self.hw['ffn2']['mult']]
        rqs_ffn_shift = [self.hw['ffn1']['shift'], self.hw['ffn2']['shift']]
        rqs_ffn_add = [self.hw['ffn1']['add'], self.hw['ffn2']['add']]
        self._write_flat_row(rqs_ffn_mul, "RQS_FFN_MUL.txt")
        self._write_flat_row(rqs_ffn_shift, "RQS_FFN_SHIFT.txt")
        self._write_flat_row(rqs_ffn_add, "RQS_FFN_ADD.txt")
        print("  ✅ Wrote global parameter files (RQS, GELU).")

    def write_standalone_files(self):
        """Generates all verification files for the 'standalone' testbench."""
        # This carefully replicates the file generation from ITA.py
        path = self.paths['standalone']
        
        # Inputs
        self._write_standalone_tensor(self.t['Q_in'], 'Q')
        self._write_standalone_tensor(self.t['K_in'], 'K')
        self._write_standalone_tensor(self.t['V_in'], 'V')
        self._write_standalone_tensor(self.t['FF_in'], 'FF')

        # Weights and Biases (squeezing to remove head dimension of 1)
        for name in ['Wq', 'Wk', 'Wv', 'Wo', 'Wff', 'Wff2']:
             self._write_standalone_tensor(self.t[name].squeeze(0).T, f"{name}_0")
        for name in ['Bq', 'Bk', 'Bv', 'Bo', 'Bff', 'Bff2']:
             self._write_standalone_tensor(self.t[name].squeeze(0), f"{name}_0")

        # Intermediate Tensors
        self._write_standalone_tensor(self.t['Qp_requant'], 'Qp_0')
        self._write_standalone_tensor(self.t['Kp_requant'], 'Kp_0')
        self._write_standalone_tensor(self.t['Vp_requant'], 'Vp_0')
        self._write_standalone_tensor(self.t['A_requant'], 'A_0')
        self._write_standalone_tensor(self.t['A_partial_softmax'], 'A_soft_0')
        self._write_standalone_tensor(self.t['O_soft_requant'], 'O_soft_0')
        self._write_standalone_tensor(self.t['Out_soft_requant'], 'Out_soft_0')
        self._write_standalone_tensor(self.t['FFp_requant'], 'FFp_0')
        self._write_standalone_tensor(self.t['gelu'], 'gelu')
        self._write_standalone_tensor(self.t['FF2p_requant'], 'FF2p_0')

        # Create empty placeholder files as in ITA.py to prevent testbench errors
        open(os.path.join(path, "preactivation.txt"), 'w').close()
        open(os.path.join(path, "relu.txt"), 'w').close()
        print("  ✅ Wrote all standalone verification files.")

    def write_hwpe_files(self):
        """Generates all data files for the 'hwpe' testbench."""
        # Create an empty mem.txt file first, as in ITA.py
        open(os.path.join(self.paths['hwpe'], 'mem.txt'), 'w').close()

        hwpe_map = {
            'Q': self.t['Q_in'],
            'K': self.t['K_in'],
            'V': self.t['Vp_requant'].transpose(0, 2, 1), # Vp is transposed for this stage
            'QK': self.t['A_requant'],
            'A': self.t['A_partial_softmax'],
            'AV': self.t['O_soft_requant'],
            'OW': self.t['Out_soft_requant'],
            'F1': self.t['FFp_requant'],
            'F2': self.t['FF2p_requant']
        }
        for name, tensor in hwpe_map.items():
            self._write_hwpe_tensor(tensor, name)
        print("  ✅ Wrote all HWPE data files.")
            
    def write_npz_files(self):
        """Writes the .npz summary files."""
        path = self.paths['base']
        np.savez(f"{path}/mha.npz",
                 q=self.t['Q_in'].squeeze(0), k=self.t['K_in'].squeeze(0),
                 w1=self.t['Wq'], b1=self.t['Bq'],
                 w2=self.t['Wk'], b2=self.t['Bk'],
                 w3=self.t['Wv'], b3=self.t['Bv'],
                 w4=self.t['Wo'], b4=self.t['Bo'],
                 o=self.t['Out_soft_requant'].squeeze(0))
        np.savez(f"{path}/inputs.npz", inputs=self.t['Q_in'].squeeze(0))
        np.savez(f"{path}/outputs.npz", outputs=self.t['FF2p_requant'].squeeze(0))
        print("  ✅ Wrote summary .npz files.")

    def write_all(self):
        """Creates directories and writes all required files."""
        self.create_directories()
        self.write_global_params()
        self.write_standalone_files()
        self.write_hwpe_files()
        self.write_npz_files()
        # Note: mempool and snitch-cluster file generation would be added here
        # if their complex, specific layouts are required.
        print("--- ✅ All Files Exported Successfully! ---")

# ==============================================================================
# FUNCTION 3: PYTORCH-TO-HARDWARE PARAMETER TRANSLATION
# This function is the critical bridge between the learned PyTorch scales and
# the integer parameters required by the hardware.
# ==============================================================================       
def translate_scales_to_hw_params(model, block_idx):
    """
    Extracts learned scales AND zero-points from a converted QAT model and 
    translates them into a complete hardware parameter dictionary for a specific block.
    """
    print(f"--- ⚙️  Translating PyTorch scales & zero-points to HW params for Block {block_idx} ---")
    hw_params = {}
    attn = model.attention_blocks[block_idx]
    ffn = model.ffn_blocks[block_idx]

    # --- Attention Block Parameter Extraction ---
    # The scale of the input tensor to the attention block.
    s_in_attn = model.quant_attention[block_idx].scale.item()

    # Step 1: Q Projection
    s_eff_q = (s_in_attn * attn.q_proj.weight().q_scale()) / attn.q_proj.scale
    mult_q, shift_q = calculate_multiplier_shift(s_eff_q, 16)
    hw_params['q_proj'] = {'mult': mult_q, 'shift': shift_q, 'add': attn.q_proj.zero_point}

    # Step 2: K Projection
    s_eff_k = (s_in_attn * attn.k_proj.weight().q_scale()) / attn.k_proj.scale
    mult_k, shift_k = calculate_multiplier_shift(s_eff_k, 16)
    hw_params['k_proj'] = {'mult': mult_k, 'shift': shift_k, 'add': attn.k_proj.zero_point}

    # Step 3: V Projection
    s_eff_v = (s_in_attn * attn.v_proj.weight().q_scale()) / attn.v_proj.scale
    mult_v, shift_v = calculate_multiplier_shift(s_eff_v, 16)
    hw_params['v_proj'] = {'mult': mult_v, 'shift': shift_v, 'add': attn.v_proj.zero_point}

    # Step 4: QK Matmul (Attention Scores)
    # Input scales are the outputs of the Q and K projections. Output scale is from the matmul op.
    s_q = attn.q_proj.scale
    s_k = attn.k_proj.scale
    s_a = attn.matmul_qk.scale
    s_eff_qk = (s_q * s_k) / s_a
    mult_qk, shift_qk = calculate_multiplier_shift(s_eff_qk, 15) # Note the different bit-width from ITA.py
    hw_params['qk_matmul'] = {'mult': mult_qk, 'shift': shift_qk, 'add': attn.matmul_qk.zero_point}

    # Step 5: AV Matmul (Context Vector)
    # Input scales are the softmax output (which has the same scale as its input, s_a) and the V projection output.
    s_v = attn.v_proj.scale
    s_o_soft = attn.matmul_av.scale
    s_eff_av = (s_a * s_v) / s_o_soft
    mult_av, shift_av = calculate_multiplier_shift(s_eff_av, 11) # Note the different bit-width from ITA.py
    hw_params['av_matmul'] = {'mult': mult_av, 'shift': shift_av, 'add': attn.matmul_av.zero_point}
    
    # Step 6: Output Projection
    # Input scale is the output of the AV matmul.
    s_eff_o = (s_o_soft * attn.out_proj.weight().q_scale()) / attn.out_proj.scale
    mult_o, shift_o = calculate_multiplier_shift(s_eff_o, 16)
    hw_params['out_proj'] = {'mult': mult_o, 'shift': shift_o, 'add': attn.out_proj.zero_point}

    # --- Feed-Forward Network Parameter Extraction ---
    s_in_ffn = model.quant_ffn[block_idx].scale.item()
    
    # Step 7: FFN1
    s_eff_ffn1 = (s_in_ffn * ffn.fc1.weight().q_scale()) / ffn.fc1.scale
    mult_ffn1, shift_ffn1 = calculate_multiplier_shift(s_eff_ffn1, 16)
    hw_params['ffn1'] = {'mult': mult_ffn1, 'shift': shift_ffn1, 'add': ffn.fc1.zero_point}

    # Step 8: FFN2
    # The input scale to FFN2 is the output scale of the activation (ReLU/GELU).
    # In QAT, this is the same as the output scale of the preceding layer, ffn.fc1.
    s_in_ffn2 = ffn.fc1.scale
    s_eff_ffn2 = (s_in_ffn2 * ffn.fc2.weight().q_scale()) / ffn.fc2.scale
    mult_ffn2, shift_ffn2 = calculate_multiplier_shift(s_eff_ffn2, 16)
    hw_params['ffn2'] = {'mult': mult_ffn2, 'shift': shift_ffn2, 'add': ffn.fc2.zero_point}

    # --- GELU Parameters (Fixed by Hardware Design) ---
    # These are not learned but are part of the hardware's fixed configuration.
    hw_params['gelu'] = {"q_1": -22, "q_b": -14, "q_c": 24, "rqs_mul": 119, "rqs_shift": 20, "rqs_add": 0}

    print("    ✅ Translation complete.")
    return hw_params

# ==============================================================================
# MAIN EXPORT SCRIPT
# ==============================================================================

def export_all_vectors(checkpoint_path):
    """Main export function."""
    print("--- 🚀 Starting Comprehensive Export ---")
    
    # --- Step 1: Load and Prepare Quantized Model ---
    # Use the EXPORT version of the model, which is designed for this purpose.
    model = ITALSTMNetVIT_Export() 

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

    
    model.qconfig = ita_symmetric_qconfig
    model.tokenizer.qconfig = None # The tokenizer runs on the CPU and is not quantized.

    # Convert the model to its final integer-only representation.
    torch.quantization.prepare_qat(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.eval()

    # Load the learned weights and quantization parameters from the training checkpoint.
    print("Loading quantized state dict into a converted model structure...")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f"✅ Loaded quantized model from {checkpoint_path}")
    
    # --- Step 2: Setup Paths and Ground Truth Input ---
    model_params = {'S': model.S, 'E': model.E, 'P': model.P, 'F': model.F, 'H': model.H}
    base_folder_name = f"data_S{model.S}_E{model.E}_P{model.P}_F{model.F}_H{model.H}_B1_Relu"
    os.makedirs(base_folder_name, exist_ok=True)
    
    # Use a fixed, real-world image to generate deterministic test vectors.
    ground_truth_image_path = "training/data/170692293306/1706922931.458.png"
    print(f"--- 🖼️  Using specific ground truth image: {ground_truth_image_path} ---")
    preprocess = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    img = Image.open(ground_truth_image_path)
    input_image = preprocess(img).unsqueeze(0)
    
    # --- Step 3: Get PyTorch Intermediates for Sanity Check ---
    # Run the data through the actual PyTorch model to get a reference baseline.
    print("--- 🔍 Getting PyTorch intermediates for all blocks ---")
    with torch.no_grad():
        pytorch_intermediates_list = model.forward_with_intermediates([input_image, None, None, None])

    # --- Step 4: Main Loop - Process Each Block ---
    x_float, H, W = model.tokenizer(input_image)

    for block_idx in range(len(model.attention_blocks)):
        print(f"\n{'='*20} Processing Block {block_idx} {'='*20}")
        
        # Create block-specific sub-directories
        block_folder_path = os.path.join(base_folder_name, f"Block_{block_idx}")
        
        # Create sub_dirs for FileWriter
        sub_dirs = {
            "base": block_folder_path,
            "hwpe": os.path.join(block_folder_path, "hwpe"),
            "mempool": os.path.join(block_folder_path, "mempool"),
            "standalone": os.path.join(block_folder_path, "standalone"),
            "snitch": os.path.join(block_folder_path, "snitch-cluster")
        }
        for path in sub_dirs.values():
            os.makedirs(path, exist_ok=True)

        # 4.1. Translate learned PyTorch scales into hardware parameters.
        hw_params = translate_scales_to_hw_params(model, block_idx)
    
        # 4.2. Extract all weights, biases, and inputs for the current block.
        tensors = {}
        attn_block = model.attention_blocks[block_idx]
        ffn_block = model.ffn_blocks[block_idx]
        
        with torch.no_grad():
            q_in_torch = model.quant_attention[block_idx](x_float)
            attn_out_torch, _ = attn_block(q_in_torch, H, W)
            attn_out_float = model.dequant_attention[block_idx](attn_out_torch)
            res1_float = model.add.add(x_float, attn_out_float)
            norm1_out_float = model.norm1_layers[block_idx](res1_float)
            ffn_in_torch = model.quant_ffn[block_idx](norm1_out_float)
        
        tensors['Q_in'] = q_in_torch.int_repr().cpu().numpy()
        tensors['K_in'], tensors['V_in'] = tensors['Q_in'].copy(), tensors['Q_in'].copy()
        tensors['FF_in'] = ffn_in_torch.int_repr().cpu().numpy()
        
        # Get input scales needed for bias conversion
        s_in_attn = model.quant_attention[block_idx].scale.item()
        s_in_ffn = model.quant_ffn[block_idx].scale.item()

        layers_to_extract = {
            'Wq': (attn_block.q_proj, s_in_attn), 'Wk': (attn_block.k_proj, s_in_attn),
            'Wv': (attn_block.v_proj, s_in_attn), 'Wo': (attn_block.out_proj, attn_block.matmul_av.scale),
            'Wff': (ffn_block.fc1, s_in_ffn), 'Wff2': (ffn_block.fc2, ffn_block.fc1.scale)
        }
        
        for prefix, (module, s_input) in layers_to_extract.items():
            # Extract 8-bit integer weight
            tensors[prefix] = module.weight().int_repr().cpu().numpy().T[np.newaxis, :, :]
            
            # Convert learned float bias to hardware-compliant int32 bias
            s_weight = module.weight().q_scale()
            bias_scale = s_input * s_weight
            float_bias = module.bias().dequantize().cpu().numpy()
            int32_bias = np.round(float_bias / bias_scale).astype(np.int32)
            tensors[prefix.replace('W', 'B')] = int32_bias[np.newaxis, :]
            
        # 4.3. Run the bit-accurate hardware simulation.
        golden_model = HardwareGoldenModel(model_params, hw_params)
        golden_model.load_parameters(tensors)
        final_tensors = golden_model.run_simulation()
        
        # 4.4. Sanity check the simulation against the PyTorch reference.
        print(f"  🔬 Sanity Check for Block {block_idx}...")
        pytorch_intermediates = pytorch_intermediates_list[block_idx]
        all_match = True
        for name, golden_tensor in final_tensors.items():
            if name in pytorch_intermediates:
                pytorch_tensor = pytorch_intermediates[name].cpu().numpy()
                mae = np.mean(np.abs(golden_tensor.astype(np.float32) - pytorch_tensor.astype(np.float32)))
                if mae > 1.001: # Allow a tolerance of 1 due to rounding differences
                    print(f"    - {name:<20} MAE = {mae:.4f} ⚠️ MISMATCH")
                    all_match = False
        print(f"  ✅ Sanity Check {'PASSED' if all_match else 'FAILED'}.")
        
        # 4.5. Write all verification files for this block.
        file_writer = FileWriter(final_tensors, hw_params, model_params, block_folder_path)
        file_writer.write_all()

        # 4.6. Update x_float to be the input for the next block.
        with torch.no_grad():
            ffn_in_torch = model.quant_ffn[block_idx](norm1_out_float) # Assuming norm1_out_float is available
            ffn_out_torch, _ = ffn_block(ffn_in_torch, H, W)
            ffn_out_float = model.dequant_ffn[block_idx](ffn_out_torch)
            res2_float = model.add.add(norm1_out_float, ffn_out_float)
            x_float = model.norm2_layers[block_idx](res2_float)
    
    # --- Step 5: Export Global Files ---
    print("\n--- Exporting global model files (ONNX)... ---")
    torch.onnx.export(model, (input_image, torch.rand(1,1), torch.rand(1,4), None), 
                      os.path.join(base_folder_name, "network.onnx"), opset_version=17, 
                      input_names=['image', 'desvel', 'quat', 'hidden'], 
                      output_names=['output', 'hidden_out'])
    
    print("\n--- ✅ All Files Exported Successfully! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch QAT model tensors for hardware verification.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the final quantized model state_dict (.pth)')
    args = parser.parse_args()
    export_all_vectors(args.checkpoint)
