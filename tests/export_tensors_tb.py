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
from models.testing.ITA_model import ITALSTMNetVIT
from models.testing.ITA_layers import ITASoftmax, ITAGELU
from third_party.vitfly_FPGA.training.train import TRAINER
from third_party.ITA_FPGA.PyITA.util import (
    write_matrix, to_hex, pack_hex_24b, pack_array_8b_to_word, pack_8b_to_word,
    write_vector_mem_hex, write_matrix_mem_hex, split_matrix, generate_matrix_mem,
    write_matrix_mem
)

# ==============================================================================
# HELPER FUNCTIONS (Adapted from PyITA)
# ==============================================================================

def calculate_multiplier_shift(effective_scale, shift_bits=31):
    """
    Calculates the integer multiplier and shift from a float scale.
    This is now a top-level function for clarity.
    """
    if effective_scale == 0:
        return 0, 0
    multiplier = int(round(effective_scale * (2**shift_bits)))
    return multiplier, shift_bits

def np_requantize(x, mult, shift, add=0):
    """Numpy implementation of the hardware requantization function."""
    x = x.astype(np.float64) * mult
    x = np.floor(x / (2**shift) + 0.5 + np.finfo(np.float32).eps) + add
    return np.clip(x, -128, 127).astype(np.int8)

def np_i_gelu(x_in, params):
    """Numpy implementation of the hardware-approximated GELU."""
    gelu_module = ITAGELU(params)
    q_1, q_b, q_c = gelu_module.q_1.item(), gelu_module.q_b.item(), gelu_module.q_c.item()
    eps_mul, eps_shift, eps_add = gelu_module.eps_mul.item(), gelu_module.eps_shift.item(), gelu_module.eps_add.item()

    def _i_poly(q, q_b, q_c):
        d = q.astype(np.int16) + q_b
        return (d * d + q_c).astype(np.int32)
    def _i_erf(q, q_b, q_c):
        q_sgn = np.sign(q)
        q_abs = np.abs(q)
        q_clipped = np.clip(q_abs, 0, -q_b)
        return q_sgn * _i_poly(q_clipped, q_b, q_c)
    def _i_gelu(q, q_1, q_b, q_c):
        q_clipped = np.clip(q, -127, 127)
        q_erf = _i_erf(q_clipped, q_b, q_c)
        return q_clipped * (q_erf + q_1)

    q_out = _i_gelu(x_in, q_1, q_b, q_c)
    return np_requantize(q_out, eps_mul, eps_shift, eps_add)

def _generate_standalone_files(tensors, path_s, params):
    """Replicates ITA.py's tiler functions to generate all standalone files."""
    print("--- ✍️  Generating all standalone verification files... ---")
    S, P, E, F, H = params['S'], params['P'], params['E'], params['F'], params['H']
    ITA_M, ITA_N = 64, 8
    
    def save_tiled(tensor, filename):
        with open(os.path.join(path_s, filename), 'w') as f:
            np.savetxt(f, tensor, fmt='%d')

    def tile_and_save_qk(q_in, w, b, out_q, name_in, name_w, name_b, name_out):
        w_t = np.transpose(w, (0, 2, 1))
        b_broadcast = np.tile(b, [1, S, 1])
        input_tiled = split_matrix(q_in[0], (ITA_M, ITA_M), flatten=False)
        input_tiled = np.tile(input_tiled, [1, 1, ITA_M // ITA_N, 1])
        input_tiled = np.tile(input_tiled, [1, P // ITA_M, 1, 1]).reshape((-1, ITA_M))
        save_tiled(input_tiled, f"{name_in}.txt")
        save_tiled(input_tiled, f"{name_out}_in_0.txt")
        save_tiled(split_matrix(w_t[0], (ITA_M, ITA_M)), f"{name_w}_0.txt")
        save_tiled(split_matrix(b_broadcast[0], (ITA_M, ITA_N)), f"{name_b}_0.txt")
        save_tiled(split_matrix(out_q[0], (ITA_M, ITA_N)), f"{name_out}_0.txt")

    tile_and_save_qk(tensors['Q_in'], tensors['Wq'], tensors['Bq'], tensors['Qp_requant'], 'Q', 'Wq', 'Bq', 'Qp')
    tile_and_save_qk(tensors['K_in'], tensors['Wk'], tensors['Bk'], tensors['Kp_requant'], 'K', 'Wk', 'Bk', 'Kp')
    
    save_tiled(split_matrix(tensors['V_in'][0], (ITA_M, ITA_M)), 'V.txt')
    save_tiled(split_matrix(np.transpose(tensors['Wv'][0],(1,0)), (ITA_M, ITA_M)), 'Wv_0.txt')
    save_tiled(split_matrix(np.tile(tensors['Bv'], [1,S,1]).transpose(0,2,1)[0], (ITA_M, ITA_N)), 'Bv_0.txt')
    save_tiled(split_matrix(tensors['Vp_requant'][0].T, (ITA_M, ITA_N)), 'Vp_0.txt')

    def tile_and_save_av(q, k, out, name_in1, name_in2, name_out):
        num_tiles_y = k.shape[2] // ITA_M
        input_tiled = split_matrix(q[0], (ITA_M, ITA_M), flatten=False)
        input_tiled = np.tile(input_tiled, [1, 1, ITA_M // ITA_N, 1])
        input_tiled = np.tile(input_tiled, [1, num_tiles_y, 1, 1]).reshape((-1, ITA_M))
        save_tiled(input_tiled, f"{name_in1}_in_0.txt")
        save_tiled(split_matrix(k[0], (ITA_M, ITA_M)), f"{name_in2}_in_0.txt")
        save_tiled(split_matrix(out[0], (ITA_M, ITA_N)), f"{name_out}_0.txt")
        
    tile_and_save_av(tensors['Qp_requant'], tensors['Kp_requant'], tensors['A_requant'], 'Qp', 'Kp', 'A')
    tile_and_save_av(tensors['A_requant'], tensors['Vp_requant'].transpose(0,2,1), tensors['O_soft_requant'], 'A_stream_soft', 'Vp', 'O_soft')
    
    save_tiled(np.tile(tensors['A_partial_softmax'][0], [ITA_M // ITA_N, 1]), "A_soft_in.txt")
    save_tiled(split_matrix(tensors['A_partial_softmax'][0], (ITA_M, ITA_N)), "A_soft_0.txt")

    def tile_and_save_out(o_in, w, b, out, name_in, name_w, name_b, name_out):
        w_t = np.transpose(w, (0, 2, 1))
        b_broadcast = np.tile(b, [1, S, 1])
        input_tiled = split_matrix(o_in[0], (ITA_M, ITA_M), flatten=False)
        input_tiled = np.tile(input_tiled, [1, 1, ITA_M // ITA_N, 1])
        input_tiled = np.tile(input_tiled, [1, E // ITA_M, 1, 1]).reshape((-1, ITA_M))
        save_tiled(input_tiled, f"{name_in}_in_0.txt")
        save_tiled(split_matrix(w_t[0], (ITA_M, ITA_M)), f"{name_w}_0.txt")
        save_tiled(split_matrix(b_broadcast[0], (ITA_M, ITA_N)), f"{name_b}_0.txt")
        save_tiled(split_matrix(out[0], (ITA_M, ITA_N)), f"{name_out}_0.txt")

    tile_and_save_out(tensors['O_soft_requant'], tensors['Wo'], tensors['Bo'], tensors['Out_soft_requant'], 'O_soft', 'Wo', 'Bo', 'Out_soft')
    tile_and_save_qk(tensors['FF_in'], tensors['Wff'], tensors['Bff'], tensors['FFp_requant'], 'FF', 'Wff', 'Bff', 'FFp')
    tile_and_save_out(tensors['gelu'], tensors['Wff2'], tensors['Bff2'], tensors['FF2p_requant'], 'FFp', 'Wff2', 'Bff2', 'FF2p')

    np.savetxt(f"{path_s}/gelu.txt", split_matrix(tensors['gelu'][0], (ITA_M, ITA_N)), fmt='%d')
    open(f"{path_s}/preactivation.txt", 'w').close()
    open(f"{path_s}/relu.txt", 'w').close()

def _export_snitch_cluster_files(tensors, path, params):
    """Generates the mem_snitch_cluster.h file."""
    print("--- ✍️  Generating snitch-cluster C header file... ---")
    S, P, E, F, H = params['S'], params['P'], params['E'], params['F'], params['H']
    
    def generate_c_array(arr, name, c_type="int8_t"):
        if arr.ndim > 2:
             return f"const {c_type} {name}[{H}][{arr[0].size}] = {{\n" + \
                    ",\n".join([f"{{\n{generate_matrix_mem(a)}\n}}" for a in arr]) + \
                    "\n}};\n"
        return f"const {c_type} {name}[{arr.size}] = {{\n{generate_matrix_mem(arr)}\n}};\n"

    with open(os.path.join(path, "mem_snitch_cluster.h"), "w") as f:
        f.write("/* This file is automatically generated. Do not edit. */\n\n// clang-format off\n")
        f.write(generate_c_array(split_matrix(tensors['Q_in'][0], (64,64)), "input_q"))
        f.write(generate_c_array(split_matrix(tensors['K_in'][0], (64,64)), "input_k"))
        f.write(generate_c_array([split_matrix(w.T, (64,64)) for w in tensors['Wq']], "input_Wq"))
        f.write(generate_c_array([split_matrix(w.T, (64,64)) for w in tensors['Wk']], "input_Wk"))
        f.write(generate_c_array([split_matrix(w.T, (64,64)) for w in tensors['Wv']], "input_Wv"))
        f.write(generate_c_array([split_matrix(w.T, (64,64)) for w in tensors['Wo']], "input_Wo"))
        f.write(generate_c_array(tensors['Bq'], "input_Bq", "int32_t"))
        f.write(generate_c_array(tensors['Bk'], "input_Bk", "int32_t"))
        f.write(generate_c_array(tensors['Bv'], "input_Bv", "int32_t"))
        f.write(generate_c_array(tensors['Bo'], "input_Bo", "int32_t"))
        f.write(generate_c_array([split_matrix(g, (64,64)) for g in tensors['Out_soft_requant']], "golden_output"))
        f.write(f"\n#define HEADS {H}\n")
        f.write(f"#define SEQUENCE_LENGTH {S}\n")
        f.write(f"#define EMBEDDING_SPACE {E}\n")
        f.write(f"#define PROJECTION_SPACE {P}\n")
        f.write("\n// clang-format on\n")

def _export_mempool_files(tensors, path, params):
    """Generates the mem.c file with a specific memory layout."""
    print("--- ✍️  Generating mempool C source file... ---")
    S, E, P, H = params['S'], params['E'], params['P'], params['H']
    
    with open(os.path.join(path, 'mem.c'), "w") as f:
        f.write("/* This file is automatically generated. Do not edit. */\n\n#include <stdint.h>\n\n// clang-format off\n")
        
        # In this simple H=1 case, we just dump the main tensors
        f.write(f'const int8_t inputs_0[] __attribute__((aligned(0x1000))) = {{\n')
        
        # This layout is complex and specific, mimicking ITA.py
        # It concatenates tensors in a particular order.
        data_to_write = [
            tensors['Wo'][0].T, tensors['Wv'][0].T, tensors['Wk'][0].T,
            np.concatenate(np.split(tensors['Q_in'][0], 4, axis=1)),
            np.concatenate(np.split(tensors['K_in'][0], 4, axis=1)),
            np.concatenate(np.split(tensors['Wq'][0].T, 4, axis=1)),
            np.reshape(np.split(np.tile(tensors['Bo'], [1,S,1])[0], 4, axis=1), (S, E)),
            np.reshape(np.split(np.reshape(np.tile(tensors['Bv'],[1,S,1])[0].T, (P,S)), 4, axis=1), (P, S)),
            np.reshape(np.split(np.tile(tensors['Bk'],[1,S,1])[0], 4, axis=1), (S, P)),
            np.reshape(np.split(np.tile(tensors['Bq'],[1,S,1])[0], 4, axis=1), (S, P)),
        ]
        
        for i, arr in enumerate(data_to_write):
            f.write(generate_matrix_mem(arr))
            if i < len(data_to_write) - 1:
                f.write(',\n')
        
        f.write('\n};\n\n// clang-format on\n')
        
def extract_and_translate_scales(model, block_idx):
    """
    Extracts scales from a QAT model and translates them to HW params.
    CORRECTED to work with the ModuleList of quantization stubs.
    """
    print(f"--- ⚙️  Translating scales for Block {block_idx} to HW Params ---")
    hw_params = {}
    
    # --- Correctly select the modules for the specified block ---
    attn = model.attention_blocks[block_idx]
    ffn = model.ffn_blocks[block_idx]
    quant_attn = model.quant_attention[block_idx]
    dequant_attn = model.dequant_attention[block_idx]
    quant_ffn = model.quant_ffn[block_idx]
    dequant_ffn = model.dequant_ffn[block_idx]

    # --- Attention Block Parameter Calculation ---
    # For quantized nn.Linear, the output scale is attached directly to the module itself.
    hw_params['mq'], hw_params['sq'] = calculate_multiplier_shift((quant_attn.scale * attn.q_proj.weight().q_scale()) / attn.q_proj.scale, 16)
    hw_params['mk'], hw_params['sk'] = calculate_multiplier_shift((quant_attn.scale * attn.k_proj.weight().q_scale()) / attn.k_proj.scale, 16)
    hw_params['mv'], hw_params['sv'] = calculate_multiplier_shift((quant_attn.scale * attn.v_proj.weight().q_scale()) / attn.v_proj.scale, 16)
    
    qkv_out_scale = attn.q_proj.scale # The scale of Qp, Kp, and Vp are the same
    
    # The scale of the attention logits (A_requant) is the scale of the custom ITASelfAttention module's output before the final projection.
    # PyTorch attaches this scale to the module itself after conversion.
    hw_params['ma'], hw_params['sa'] = calculate_multiplier_shift((qkv_out_scale * qkv_out_scale) / attn.scale, 15)
    
    # The scale of the context vector (O_soft_requant) is the scale of the final output projection layer's input.
    hw_params['mav'], hw_params['sav'] = calculate_multiplier_shift((attn.scale * qkv_out_scale) / attn.out_proj.scale, 11)
    
    # The scale of the final attention output (Out_soft_requant)
    hw_params['mo'], hw_params['so'] = calculate_multiplier_shift((attn.out_proj.scale * attn.out_proj.weight().q_scale()) / dequant_attn.scale, 16)
    
    # --- FFN Block Parameter Calculation ---
    hw_params['m_ff1'], hw_params['s_ff1'] = calculate_multiplier_shift((quant_ffn.scale * ffn.fc1.weight().q_scale()) / ffn.fc1.scale, 16)
    hw_params['m_ff2'], hw_params['s_ff2'] = calculate_multiplier_shift((ffn.fc1.scale * ffn.fc2.weight().q_scale()) / dequant_ffn.scale, 16)
    
    # --- GELU and other fixed parameters ---
    hw_params.update({"q_1": -22, "q_b": -80, "q_c": 1585, "gelu_rqs_mul": 103, "gelu_rqs_shift": 17, "gelu_rqs_add": 0})
    
    return hw_params

# ==============================================================================
# MAIN EXPORT SCRIPT
# ==============================================================================

def export_all_vectors(checkpoint_path):
    """Main export function."""
    print("--- 🚀 Starting Comprehensive Export ---")
    
    # 1. LOAD MODEL
    model = ITALSTMNetVIT(params={}, qat_mode=True) 

    # IMPORTANT: Use the exact same qconfig you used for training. 
    # This could be the symmetric one we discussed or the default.
    ita_symmetric_qconfig = QConfig(
        activation=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric, # Enforces zero_point = 0
            reduce_range=False
        ),
        weight=FusedMovingAvgObsFakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, # Enforces zero_point = 0
            reduce_range=False
        )
    )

    
    model.qconfig = ita_symmetric_qconfig
    model.tokenizer.qconfig = None

    # Prepare and convert the empty model. This step gives it the final 
    # quantized structure (e.g., nn.quantized.Linear layers).
    torch.quantization.prepare_qat(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.eval()

    # NOW, the model has the correct structure to accept the quantized state dict.
    print("Loading quantized state dict into a converted model structure...")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f"✅ Loaded quantized model from {checkpoint_path}")
    
    # 2. SETUP BASE DIRECTORY & GET GROUND TRUTH IMAGE
    model_params = {'S': model.S, 'E': model.E, 'P': model.P, 'F': model.F, 'H': model.H}
    base_folder_name = f"data_S{model.S}_E{model.E}_P{model.P}_F{model.F}_H{model.H}_B1_Relu"
    os.makedirs(base_folder_name, exist_ok=True)
    
    ground_truth_image_path = "training/data/170692293306/1706922931.458.png"
    print(f"--- 🖼️  Using specific ground truth image: {ground_truth_image_path} ---")
    preprocess = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    img = Image.open(ground_truth_image_path)
    input_image = preprocess(img).unsqueeze(0)
    
    # 3. GET PYTORCH INTERMEDIATES FOR ALL BLOCKS (for Sanity Check)
    print("--- 🔍 Getting PyTorch intermediates for all blocks ---")
    with torch.no_grad():
        pytorch_intermediates_list = model.forward_with_intermediates([input_image, None, None, None])

    # 4. MAIN LOOP: PROCESS AND EXPORT VECTORS FOR EACH BLOCK
    x_float, H, W = model.tokenizer(input_image) # Initial input to the first block

    for block_idx in range(len(model.attention_blocks)):
        print(f"\nProcessing and exporting for Block {block_idx}...")
        
        # 4.1. Create block-specific sub-directories
        block_folder_path = os.path.join(base_folder_name, f"Block_{block_idx}")
        sub_dirs = {name: os.path.join(block_folder_path, name) for name in ["hwpe", "mempool", "snitch-cluster", "standalone"]}
        for path in sub_dirs.values(): os.makedirs(path, exist_ok=True)
        print(f"  ✅ Created directory structure at: {block_folder_path}")
    
        hw_params = extract_and_translate_scales(model, block_idx)
    
        # 4.3. Get inputs and weights for this block
        tensors = {}
        attn_block, ffn_block = model.attention_blocks[block_idx], model.ffn_blocks[block_idx]
        with torch.no_grad():
            q_in_torch = model.quant_attention[block_idx](x_float)
            attn_out_float = model.dequant_attention[block_idx](attn_block(q_in_torch, H, W))
            norm1_out_float = model.norm1_layers[block_idx](model.add.add(x_float, attn_out_float))
            ffn_in_torch = model.quant_ffn[block_idx](norm1_out_float)
        
        tensors['Q_in'] = q_in_torch.int_repr().cpu().numpy()
        tensors['K_in'], tensors['V_in'] = tensors['Q_in'].copy(), tensors['Q_in'].copy()
        tensors['FF_in'] = ffn_in_torch.int_repr().cpu().numpy()
        
        for prefix, module in [('Wq', attn_block.q_proj), ('Wk', attn_block.k_proj), ('Wv', attn_block.v_proj), ('Wo', attn_block.out_proj), ('Wff', ffn_block.fc1), ('Wff2', ffn_block.fc2)]:
            tensors[prefix] = module.weight().int_repr().cpu().numpy().T[np.newaxis, :, :]
            tensors[prefix.replace('W','B')] = module.bias().int().cpu().numpy()[np.newaxis, :]

        # 4.4. Golden Model Recalculation
        print(f"  ✨ Recomputing all intermediates for Block {block_idx}...")
        tensors['Qp'] = np.matmul(tensors['Q_in'], tensors['Wq']) + np.tile(tensors['Bq'], [1, model.S, 1])
        tensors['Qp_requant'] = np.array([np_requantize(t, hw_params['mq'], hw_params['sq']) for t in tensors['Qp']])
        tensors['Kp'] = np.matmul(tensors['K_in'], tensors['Wk']) + np.tile(tensors['Bk'], [1, model.S, 1])
        tensors['Kp_requant'] = np.array([np_requantize(t, hw_params['mk'], hw_params['sk']) for t in tensors['Kp']])
        tensors['Vp'] = np.matmul(tensors['V_in'], tensors['Wv']) + np.tile(tensors['Bv'], [1, model.S, 1])
        tensors['Vp_requant'] = np.array([np_requantize(t, hw_params['mv'], hw_params['sv']) for t in tensors['Vp']])
        A_matmul = np.matmul(tensors['Qp_requant'].astype(np.float32), tensors['Kp_requant'].transpose(0,2,1).astype(np.float32))
        tensors['A_requant'] = np.array([np_requantize(t, hw_params['ma'], hw_params['sa']) for t in A_matmul])
        A_requant_torch = torch.from_numpy(tensors['A_requant'])
        tensors['A_partial_softmax'] = ITASoftmax()(A_requant_torch).cpu().numpy()
        O_soft_matmul = np.matmul(tensors['A_partial_softmax'].astype(np.float32), tensors['Vp_requant'].astype(np.float32))
        tensors['O_soft_requant'] = np.array([np_requantize(t, hw_params['mav'], hw_params['sav']) for t in O_soft_matmul])
        Out_soft_matmul = np.matmul(tensors['O_soft_requant'], tensors['Wo']) + np.tile(tensors['Bo'], [1, model.S, 1])
        tensors['Out_soft_requant'] = np.array([np_requantize(t, hw_params['mo'], hw_params['so']) for t in Out_soft_matmul])
        tensors['FFp'] = np.matmul(tensors['FF_in'], tensors['Wff']) + np.tile(tensors['Bff'], [1, model.S, 1])
        tensors['FFp_requant'] = np.array([np_requantize(t, hw_params['m_ff1'], hw_params['s_ff1']) for t in tensors['FFp']])
        tensors['gelu'] = np.array([np_i_gelu(t, hw_params) for t in tensors['FFp_requant']])
        tensors['FF2p'] = np.matmul(tensors['gelu'], tensors['Wff2']) + np.tile(tensors['Bff2'], [1, model.S, 1])
        tensors['FF2p_requant'] = np.array([np_requantize(t, hw_params['m_ff2'], hw_params['s_ff2']) for t in tensors['FF2p']])
        print(f"  ✅ Golden model tensors recomputed for Block {block_idx}.")

        # 4.5. Sanity Check
        print(f"  🔬 Sanity Check for Block {block_idx}...")
        pytorch_intermediates_np = {key: val.cpu().numpy().squeeze() for key, val in pytorch_intermediates_list[block_idx].items()}
        tensors_to_compare = ['Q_in','Out_soft_requant','FF_in','FF2p_requant','Qp_requant','Kp_requant','Vp_requant','A_requant','A_partial_softmax','O_soft_requant','FFp_requant','gelu']
        all_match = True
        for name in tensors_to_compare:
            mae = np.mean(np.abs(tensors[name].squeeze().astype(np.float32) - pytorch_intermediates_np[name].astype(np.float32)))
            if mae > 0.01:
                print(f"    - {name:<20} MAE = {mae:.4f} ⚠️")
                all_match = False
        if all_match: print("  ✅ Sanity Check: PASSED.")
        else: print("  ⚠️ Sanity Check: FAILED. Drift detected.")

        # 6. WRITE ALL FILES IN ITA.PY FORMAT
        print(f"--- 💾 Writing all files to ./{block_folder_path} ---")
        def write_flat_row(data, filename): np.savetxt(filename, np.array([data]), fmt='%d', delimiter=' ')
        write_flat_row([hw_params['q_b']], f"{block_folder_path}/GELU_B.txt")
        np.savetxt(f"{block_folder_path}/GELU_C.txt", [[1585]], fmt='%d')
        np.savetxt(f"{block_folder_path}/GELU_ONE.txt", [[-22]], fmt='%d')
        rqs_attn_mul = [hw_params['mq'], hw_params['mk'], hw_params['mv'], hw_params['ma'], hw_params['mav'], hw_params['mo'], 0]
        rqs_attn_shift = [16, 16, 16, 15, 11, 16, 7]
        rqs_ffn_mul = [hw_params['m_ff1'], hw_params['m_ff2']]
        rqs_ffn_shift = [16, 16]
        write_flat_row(rqs_attn_mul, f"{block_folder_path}/RQS_ATTN_MUL.txt")
        write_flat_row(rqs_attn_shift, f"{block_folder_path}/RQS_ATTN_SHIFT.txt")
        write_flat_row(np.zeros_like(rqs_attn_mul), f"{block_folder_path}/RQS_ATTN_ADD.txt")
        write_flat_row(rqs_ffn_mul, f"{block_folder_path}/RQS_FFN_MUL.txt")
        write_flat_row(rqs_ffn_shift, f"{block_folder_path}/RQS_FFN_SHIFT.txt")
        write_flat_row(np.zeros_like(rqs_ffn_mul), f"{block_folder_path}/RQS_FFN_ADD.txt")
        
        _generate_standalone_files(tensors, sub_dirs['standalone'], model_params)
        def write_hwpe_tensor(tensor_np, filename):
            tiles = split_matrix(tensor_np.squeeze(), block_shape=(64, 64))
            packed_hex = pack_array_8b_to_word(tiles, hex_string=False)
            filepath = os.path.join(sub_dirs['hwpe'], f'{filename}.txt')
            if os.path.exists(filepath): os.remove(filepath)
            write_matrix_mem_hex(packed_hex, filename, sub_dirs['hwpe'])
        for name, t in {'Q':'Q_in','K':'K_in','V':'Vp_requant','QK':'A_requant','A':'A_partial_softmax','AV':'O_soft_requant','OW':'Out_soft_requant','F1':'FFp_requant','F2':'FF2p_requant'}.items():
            write_hwpe_tensor(tensors[t] if t != 'Vp_requant' else tensors[t].transpose(0,2,1), name)
        with open(os.path.join(sub_dirs['hwpe'], 'mem.txt'), 'w') as f: f.write("# Minimal mem file for HWPE simulation.\n")

        _export_snitch_cluster_files(tensors, sub_dirs['snitch-cluster'], model_params)
        _export_mempool_files(tensors, sub_dirs['mempool'], model_params)
        
        np.savez(f"{block_folder_path}/mha.npz", q=tensors['Q_in'].squeeze(), k=tensors['K_in'].squeeze(), o=tensors['Out_soft_requant'].squeeze())
        np.savez(f"{block_folder_path}/inputs.npz", inputs=tensors['Q_in'].squeeze())
        np.savez(f"{block_folder_path}/outputs.npz", outputs=tensors['Out_soft_requant'].squeeze())
        torch.onnx.export(model, (input_image, torch.rand(1,1), torch.rand(1,4), None), f"{block_folder_path}/network.onnx", opset_version=13, input_names=['image', 'desvel', 'quat', 'hidden'], output_names=['output', 'hidden_out'])
        
        # 4.7. Update x_float for the next block's input
        with torch.no_grad():
            ffn_out_float = model.dequant_ffn[block_idx](ffn_block(ffn_in_torch, H, W))
            res2_float = model.add.add(norm1_out_float, ffn_out_float)
            x_float = model.norm2_layers[block_idx](res2_float)
    
    # 5. EXPORT GLOBAL FILES (ONNX)
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
