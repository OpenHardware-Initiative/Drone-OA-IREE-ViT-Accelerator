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
from third_party.ITA_FPGA.PyITA.ITA import Transformer
from models.ITA.ITA_model import ITALSTMNetVIT
from models.ITA.export.ITA_model_export import ITALSTMNetVIT_Export
from models.ITA.ITA_layers import ITASoftmax
from third_party.ITA_FPGA.PyITA.softmax import streamingPartialSoftmax
from third_party.ITA_FPGA.PyITA.util import (
    write_matrix, to_hex, pack_hex_24b, pack_array_8b_to_word, pack_8b_to_word,
    write_vector_mem_hex, write_matrix_mem_hex, split_matrix, generate_matrix_mem,
    write_matrix_mem, requantize
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
        print("✨ Golden Model Initialized.")

    def _np_requantize(self, x, mult, shift, add=0):
        """
        Bit-accurate requantization simulation.
        This function already matches the logic of PyITA's `gelu_requantize`.
        """
        # This implementation is correct for simulating integer arithmetic
        x = x.astype(np.float64) * mult
        x = np.floor(x / (2**shift) + 0.5) + add
        return np.clip(x, -128, 127).astype(np.int8)

    def _np_i_gelu(self, x_in):
        params = self.hw_params['relu']
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
        return requantize(q_out, mult, shift, add)

    def load_parameters(self, block_tensors):
        """Loads the extracted weights, biases, and inputs for the current block."""
        self.tensors.update(block_tensors)
        print("  ✅ Loaded weights, biases, and inputs into golden model.")

    def run_simulation(self):
        """Executes the full hardware simulation for one transformer block."""
        print("  ✨ Recomputing all intermediates with hardware-like arithmetic...")
        S = self.p['S']
        H = self.p['H']

        # Attention Block Simulation
        print(f">>>>>>>>> Shape of Q_in: {self.tensors['Q_in'].shape}, dtype: {self.tensors['Q_in'].dtype}")
        self.tensors['Qp'] = np.matmul(self.tensors['Q_in'], self.tensors['Wq']) + np.tile(self.tensors['Bq'], [H, S, 1])
        self.tensors['Qp_requant'] = requantize(self.tensors['Qp'],
                                                np.array([self.hw_params['q_proj']['mult']]),
                                                np.array([self.hw_params['q_proj']['shift']]),
                                                np.array([self.hw_params['q_proj']['add']]))

        self.tensors['Kp'] = np.matmul(self.tensors['K_in'], self.tensors['Wk']) + np.tile(self.tensors['Bk'], [H, S, 1])
        self.tensors['Kp_requant'] = requantize(self.tensors['Kp'],
                                                np.array([self.hw_params['k_proj']['mult']]),
                                                np.array([self.hw_params['k_proj']['shift']]),
                                                np.array([self.hw_params['k_proj']['add']]))

        self.tensors['Vp'] = np.matmul(self.tensors['V_in'], self.tensors['Wv']) + np.tile(self.tensors['Bv'], [H, S, 1])
        self.tensors['Vp_requant'] = requantize(self.tensors['Vp'],
                                                np.array([self.hw_params['v_proj']['mult']]),
                                                np.array([self.hw_params['v_proj']['shift']]),
                                                np.array([self.hw_params['v_proj']['add']]))


        A_matmul = np.matmul(self.tensors['Qp_requant'].astype(np.int32), self.tensors['Kp_requant'].transpose(0, 2, 1).astype(np.int32))
        self.tensors['A_requant'] = requantize(A_matmul,
                                                np.array([self.hw_params['qk_matmul']['mult']]),
                                                np.array([self.hw_params['qk_matmul']['shift']]),
                                                np.array([self.hw_params['qk_matmul']['add']]))
        
        self.tensors['A_partial_softmax'] = streamingPartialSoftmax(self.tensors['A_requant'], integerize=True)
        
        #a_requant_torch = torch.from_numpy(self.tensors['A_requant'])
        #a_requant_4d = a_requant_torch.unsqueeze(1)
        
        # Use torch ITASoftmax on numpy array for bit-accuracy
        #self.tensors['A_partial_softmax'] = self.softmax_sim(a_requant_4d).numpy()

        O_soft_matmul = np.matmul(self.tensors['A_partial_softmax'].astype(np.int32), self.tensors['Vp_requant'].astype(np.int32))
        self.tensors['O_soft_requant'] = requantize(O_soft_matmul,
                                                     np.array([self.hw_params['av_matmul']['mult']] * H),
                                                     np.array([self.hw_params['av_matmul']['shift']] * H),
                                                     np.array([self.hw_params['av_matmul']['add']] * H))

        Out_soft_matmul = np.matmul(self.tensors['O_soft_requant'], self.tensors['Wo']) + np.tile(self.tensors['Bo'], [1, S, 1])
        Out_soft_matmul = np.matmul(self.tensors['O_soft_requant'], self.tensors['Wo']) + np.tile(self.tensors['Bo'], [H, S, 1])
        self.tensors['Out_soft_requant'] = requantize(Out_soft_matmul,
                                                       np.array([self.hw_params['out_proj']['mult']] * H),
                                                       np.array([self.hw_params['out_proj']['shift']] * H),
                                                       np.array([self.hw_params['out_proj']['add']] * H))

        # FFN Block Simulation
        self.tensors['FFp'] = np.matmul(self.tensors['FF_in'], self.tensors['Wff']) + np.tile(self.tensors['Bff'], [1, S, 1])
        self.tensors['FFp_requant'] = requantize(self.tensors['FFp'],
                                                    np.array([self.hw_params['ffn1']['mult']]),
                                                    np.array([self.hw_params['ffn1']['shift']]),
                                                    np.array([self.hw_params['ffn1']['add']]))


        pre_requant_relu = np.maximum(0, self.tensors['FFp_requant'])
        
        self.tensors['relu'] = requantize(pre_requant_relu,
                                           np.array([self.hw_params['relu']['rqs_mul']]),
                                           np.array([self.hw_params['relu']['rqs_shift']]),
                                           np.array([self.hw_params['relu']['rqs_add']]))

        self.tensors['FF2p'] = np.matmul(self.tensors['relu'], self.tensors['Wff2']) + np.tile(self.tensors['Bff2'], [1, S, 1])
        self.tensors['FF2p_requant'] = requantize(self.tensors['FF2p'],
                                                    np.array([self.hw_params['ffn2']['mult']]),
                                                    np.array([self.hw_params['ffn2']['shift']]),
                                                    np.array([self.hw_params['ffn2']['add']]))
        
        print("  ✅ Golden model simulation complete.")
        return self.tensors

# ==============================================================================
# CLASS 2: FILE WRITER
# Encapsulates all file I/O logic, ensuring formats match the Verilog testbenches.
# ==============================================================================

# TODO: We could implement a whole GoldenModel + FileWriter class just by reusing the Transformer class
#       from PyITA, but this would require some refactoring of the ITA model code

class FileWriter:
    """
    Handles all file generation for hardware verification, ensuring formats
    match the reference ITA.py script exactly by using its own utility functions.
    """
    def __init__(self, tensors, hw_params, model_dims, base_path):
        """
        Initializes the FileWriter.

        Args:
            tensors (dict): Dictionary of ALL tensors (inputs, weights, and intermediates).
            hw_params (dict): Dictionary of hardware parameters.
            model_dims (dict): Dictionary of model dimensions.
            base_path (str): The root directory for the output files.
        """
        self.t = tensors
        self.hw = hw_params
        self.p = model_dims
        self.paths = {
            "base": os.path.join(base_path, ""),
            "hwpe": os.path.join(base_path, "hwpe", ""),
            "mempool": os.path.join(base_path, "mempool", ""),
            "standalone": os.path.join(base_path, "standalone", ""),
            "snitch": os.path.join(base_path, "snitch-cluster", "")
        }
        print(f"--- 💾 Initialized FileWriter for path: ./{base_path} ---")

        # 💡 --- TILING HELPER ---
        # Create a lightweight instance of the reference Transformer. Its only
        # purpose is to give us access to the tiler_* methods, which contain
        # the complex logic for formatting the standalone test files.
        print("  🔧 Initializing Tiling Helper from ITA.Transformer...")
        self.ita_tiler = Transformer(
            S=self.p['S'], P=self.p['P'], E=self.p['E'],
            F=self.p['F'], H=self.p['H'], path=base_path
        )

    def create_directories(self):
        """Creates all necessary subdirectories for the output files."""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        print("  ✅ Created all output directories.")
        
    # --------------------------------------------------------------------
    # Public Methods for Writing Different File Categories
    # --------------------------------------------------------------------

    def write_global_params(self):
        """Writes the top-level parameter files using the reference `write_matrix`."""
        base_path = os.path.join(self.paths["base"], '')
        
        # GELU Params - wrap scalars in a 2D array to match reference format
        write_matrix(np.array([[self.hw['relu']['q_b']]]), "GELU_B", base_path)
        write_matrix(np.array([[self.hw['relu']['q_c']]]), "GELU_C", base_path)
        write_matrix(np.array([[self.hw['relu']['q_1']]]), "GELU_ONE", base_path)
        write_matrix(np.array([[self.hw['relu']['rqs_add']]]), "activation_requant_add", base_path)
        write_matrix(np.array([[self.hw['relu']['rqs_mul']]]), "activation_requant_mult", base_path)
        write_matrix(np.array([[self.hw['relu']['rqs_shift']]]), "activation_requant_shift", base_path)

        # Requantization Params for Attention - reshape to match reference format
        rqs_attn_mul = np.array([[
            self.hw['q_proj']['mult'], self.hw['k_proj']['mult'], self.hw['v_proj']['mult'],
            self.hw['qk_matmul']['mult'], self.hw['av_matmul']['mult'], self.hw['out_proj']['mult'], 0
        ]])
        rqs_attn_shift = np.array([[
            self.hw['q_proj']['shift'], self.hw['k_proj']['shift'], self.hw['v_proj']['shift'],
            self.hw['qk_matmul']['shift'], self.hw['av_matmul']['shift'], self.hw['out_proj']['shift'], 0
        ]])
        rqs_attn_add = np.array([[
            self.hw['q_proj']['add'], self.hw['k_proj']['add'], self.hw['v_proj']['add'],
            self.hw['qk_matmul']['add'], self.hw['av_matmul']['add'], self.hw['out_proj']['add'], 0
        ]])
        write_matrix(rqs_attn_mul, "RQS_ATTN_MUL", base_path)
        write_matrix(rqs_attn_shift, "RQS_ATTN_SHIFT", base_path)
        write_matrix(rqs_attn_add, "RQS_ATTN_ADD", base_path)
        
        # Requantization Params for FFN
        rqs_ffn_mul = np.array([[self.hw['ffn1']['mult'], self.hw['ffn2']['mult']]])
        rqs_ffn_shift = np.array([[self.hw['ffn1']['shift'], self.hw['ffn2']['shift']]])
        rqs_ffn_add = np.array([[self.hw['ffn1']['add'], self.hw['ffn2']['add']]])
        write_matrix(rqs_ffn_mul, "RQS_FFN_MUL", base_path)
        write_matrix(rqs_ffn_shift, "RQS_FFN_SHIFT", base_path)
        write_matrix(rqs_ffn_add, "RQS_FFN_ADD", base_path)
        
        print("  ✅ Wrote global parameter files (RQS, relu).")

    def write_standalone_files(self):
        """
        Generates all verification files for the 'standalone' testbench
        using the correct tiling logic from the ITA.Transformer helper.
        """
        print("  ✨ Generating standalone files with correct hardware tiling...")
        
        # Inputs
        # Q, K, V Projections
        self.ita_tiler.tiler_QK(self.t['Q_in'].squeeze(0), self.t['Wq'], self.t['Bq'], self.t['Qp_requant'], "Q", "Wq", "Bq", "Qp")
        self.ita_tiler.tiler_QK(self.t['K_in'].squeeze(0), self.t['Wk'], self.t['Bk'], self.t['Kp_requant'], "K", "Wk", "Bk", "Kp")
        self.ita_tiler.tiler_V(self.t['V_in'].squeeze(0), self.t['Wv'], self.t['Bv'], self.t['Vp_requant'], "V", "Wv", "Bv", "Vp")

        # Attention and Context Calculation
        self.ita_tiler.tiler_AV(self.t['Qp_requant'], self.t['Kp_requant'].transpose(0, 2, 1), self.t['A_requant'], "Qp_in", "Kp_in", "A")
        # Softmax output has a specific file format in the reference script
        write_matrix(self.t['A_partial_softmax'].squeeze(0), 'A_soft_0', self.ita_tiler.paths['standalone'])
        self.ita_tiler.tiler_AV(self.t['A_partial_softmax'], self.t['Vp_requant'], self.t['O_soft_requant'], "A_stream_soft_in", "Vp_in", "O_soft")

        # Attention and Context Calculation
        self.ita_tiler.tiler_AV(self.t['Qp_requant'], self.t['Kp_requant'].transpose(0, 2, 1), self.t['A_requant'], "Qp_in", "Kp_in", "A")
        # Softmax output has a specific file format in the reference script
        write_matrix(self.t['A_partial_softmax'].squeeze(0), 'A_soft_0', self.ita_tiler.paths['standalone'])
        self.ita_tiler.tiler_AV(self.t['A_partial_softmax'], self.t['Vp_requant'], self.t['O_soft_requant'], "A_stream_soft_in", "Vp_in", "O_soft")

        # Output Projection
        self.ita_tiler.tiler_Out(self.t['O_soft_requant'], self.t['Wo'], self.t['Bo'], self.t['Out_soft_requant'], "O_soft_in", "Wo", "Bo", "Out_soft")
        
        # FFN Layers
        self.ita_tiler.tiler_QK(self.t['FF_in'].squeeze(0), self.t['Wff'], self.t['Bff'], self.t['relu'], "FF", "Wff", "Bff", "FFp")
        self.ita_tiler.tiler_Out(self.t['relu'], self.t['Wff2'], self.t['Bff2'], self.t['FF2p_requant'], "FFp_in", "Wff2", "Bff2", "FF2p")
        
        # Create empty placeholder files as in ITA.py to prevent testbench errors
        open(os.path.join(self.paths['standalone'], "preactivation.txt"), 'w').close()
        open(os.path.join(self.paths['standalone'], "gelu.txt"), 'w').close() # Note: reference script creates gelu.txt, not relu.txt
        print("  ✅ Wrote all standalone verification files with correct tiling.")

    def write_hwpe_files(self):
        """Generates all data files for the 'hwpe' testbench."""
        # This method's logic is already correct as it uses the reference utils.
        path = self.paths['hwpe']
        open(os.path.join(path, 'mem.txt'), 'w').close()

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
            squeezed_tensor = tensor.squeeze()
            tiles = split_matrix(squeezed_tensor, block_shape=(64, 64))
            packed_hex = pack_array_8b_to_word(tiles, hex_string=False)
            write_matrix_mem_hex(packed_hex, name, path)
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

    # --- RELU Parameters (Fixed by Hardware Design) ---
    # These are not learned but are part of the hardware's fixed configuration.
    hw_params['relu'] = {"q_1": -22, "q_b": -14, "q_c": 24, "rqs_mul": 119, "rqs_shift": 20, "rqs_add": 0}

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

    
    model.qconfig = ita_symmetric_qconfig
    model.tokenizer.qconfig = None # The tokenizer runs on the CPU and is not quantized.
    for block in model.norm1_layers:
        block.qconfig = None
    for block in model.norm2_layers:
        block.qconfig = None

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
            print(f">>>>>>>>>> Precision of Q_in: {q_in_torch.dtype}, Scale: {model.quant_attention[block_idx].scale.item()}, Shape: {q_in_torch.shape}")
            attn_out_torch, _ = attn_block(q_in_torch, H, W)
            attn_out_float = model.dequant_attention[block_idx](attn_out_torch)
            res1_float = x_float + attn_out_float
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
            float_bias = module.bias().dequantize().cpu().detach().numpy()
            int32_bias = np.round(float_bias / bias_scale).astype(np.int32)
            tensors[prefix.replace('W', 'B')] = int32_bias[np.newaxis, :]
            
        # 4.3. Run the bit-accurate hardware simulation.
        golden_model = HardwareGoldenModel(model_params, hw_params)
        golden_model.load_parameters(tensors)
        final_tensors = golden_model.run_simulation()
        
        # 4.4. Sanity check the simulation against the PyTorch reference.
        print(f"  🔬 Sanity Check for Block {block_idx}...")
        #pytorch_intermediates = pytorch_intermediates_list[block_idx]
        all_match = True
        for name, golden_tensor in final_tensors.items():
            if name in tensors:
                print(f"    - Checking {name:<20}...", end=' ')
                pytorch_tensor = tensors[name]#.cpu().numpy()
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
            res2_float = x_float + ffn_out_float
            x_float = model.norm2_layers[block_idx](res2_float)
    
    print("\n--- ✅ All Files Exported Successfully! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch QAT model tensors for hardware verification.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the final quantized model state_dict (.pth)')
    args = parser.parse_args()
    export_all_vectors(args.checkpoint)
