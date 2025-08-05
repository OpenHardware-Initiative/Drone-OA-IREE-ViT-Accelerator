import numpy as np
import torch
import torch.nn.functional as F

def requantize_tensor(x, mult, shift, zp):
    x = x * mult
    x = torch.div(x, 2**shift, rounding_mode='floor')
    return torch.clamp(x + zp, -128, 127)

def ita_partial_max(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    seq_len = logits.size(-1)
    k = min(k, seq_len)
    topk_vals, topk_indices = torch.topk(logits, k, dim=-1)
    mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
    masked_logits = logits * mask
    return F.softmax(masked_logits, dim=-1)

def calculate_ita_shifts(scales, E, P, S, H):
    """
    Calculates the hardware-specific requantization shifts based on the logic
    from ITA.py, using the learned scales from the QAT process.

    Args:
        scales (dict): Dictionary containing the learned multipliers (scales)
                       from the QAT observers.
        E (int): Embedding size (must be divisible by 64).
        P (int): Projection space (must be divisible by 64).
        S (int): Sequence length (must be divisible by 64).
        H (int): Number of heads.

    Returns:
        dict: A dictionary containing the calculated integer shifts for each step.
    """
    shifts = {}

    # Steps from ITA.py's _initialize_quantization_parameters
    # Step 0, 1, 2: Q, K, V projections
    for i, key in enumerate(["mq", "mk", "mv"]):
        mult = scales.get(key, 1.0)
        max_bit_width = np.log2(mult * E * 2**9)
        shifts["s" + key[1:]] = int(np.ceil(max_bit_width) - 8 + 2)

    # Step 3: QK multiplication (Attention Logits)
    mult = scales.get("ma", 1.0)
    max_bit_width = np.log2(mult * P * 2**8)
    shifts["sa"] = int(np.ceil(max_bit_width) - 8 + 2)

    # Step 4: AV multiplication (Context Vector)
    mult = scales.get("mav", 1.0)
    max_bit_width = np.log2(mult * S * 2**5)
    shifts["sav"] = int(np.ceil(max_bit_width) - 8 + 2)
    
    # Step 5: OW (Output) projection
    mult = scales.get("mo", 1.0)
    max_bit_width = np.log2(mult * E * 2**9)
    shifts["so"] = int(np.ceil(max_bit_width) - 8 + 2)

    # Note: Feedforward shifts can be added here following the same pattern if needed.
    
    return shifts