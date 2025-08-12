import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, QConfig

# ==============================================================================
# FINAL, CORRECTED AUTOGRAD FUNCTION
# ==============================================================================

class IntegerApproximatedSoftmaxFn(Function):
    """
    Implements the hardware-aware integer-only softmax approximation.
    - Handles both float (QAT) and quantized (inference) inputs.
    - Operates along a user-specified dimension.
    - Uses a surrogate gradient for training.
    """
    @staticmethod
    def forward(ctx, x, dim):
        # x can be EITHER a float32 tensor (during QAT)
        # OR a qint8 tensor (after conversion).
        
        # --- Save context for backward pass ---
        # The surrogate gradient needs the original input and the dimension.
        ctx.dim = dim
        ctx.save_for_backward(x)

        # --- Setup constants ---
        B = 8
        range_scale = 32
        eps_max = range_scale * B / (2**B) # ~3.125
        
        # --- PATH 1: Post-Conversion (Real integer math) ---
        if x.is_quantized:
            x_int = x.int_repr().to(torch.int32)
        # --- PATH 2: During QAT Training (Simulated integer math on floats) ---
        else:
            # We must first simulate the quantization of the incoming float logits.
            # Find the dynamic range to calculate the scale.
            # Use torch.amax for clarity, which is an alias for max over all dims.
            x_max_abs = torch.amax(torch.abs(x))
            logit_scale = x_max_abs / 127.0
            # Clamp scale to avoid division by zero if input is all zeros
            logit_scale = torch.clamp(logit_scale, min=1e-8)
            # Simulate quantization: x_int = round(x / scale)
            x_int = torch.round(x / logit_scale).to(torch.int32)

        # ======================================================================
        # THE INTEGER-ONLY LOGIC - NOW USES THE 'dim' ARGUMENT
        # ======================================================================
        global_max, _ = torch.max(x_int, dim=dim, keepdim=True)
        diff = global_max - x_int
        shift = torch.floor(diff.float() * eps_max + 0.5).to(torch.int32)
        exp_numerator = (2**B) >> shift
        
        exp_partial_sum = torch.sum(exp_numerator, dim=dim, keepdim=True)
        exp_partial_sum = torch.clamp(exp_partial_sum, min=1) # Prevent division by zero
        
        # Use a higher precision intermediate for the inverse
        exp_partial_sum_inverse = torch.floor(((2**B - 1) * (2**16)) / exp_partial_sum).to(torch.int32)
        output_int = torch.floor((exp_numerator * exp_partial_sum_inverse) / (2**16)).to(torch.uint8)

        # ======================================================================
        # THE OUTPUT STAGE - MUST HANDLE BOTH QAT AND INFERENCE
        # ======================================================================
        output_scale = 1.0 / 255.0
        output_zero_point = 0
        
        # --- PATH 1: Post-Conversion -> Output a real quantized tensor ---
        if x.is_quantized:
            return torch._make_per_tensor_quantized_tensor(
                output_int, scale=output_scale, zero_point=output_zero_point
            )
        # --- PATH 2: During QAT Training -> Output a "fake quantized" float ---
        else:
            # Dequantize the integer result to a float for the next layer
            return (output_int.float() - output_zero_point) * output_scale

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim # Retrieve the dimension

        # The surrogate gradient logic is the same: it always operates on floats.
        if x.is_quantized:
            x_float = x.dequantize()
        else:
            x_float = x

        with torch.enable_grad():
            x_float = x_float.detach().requires_grad_(True)
            output_float_softmax = torch.softmax(x_float, dim=dim)
            output_float_softmax.backward(gradient=grad_output)
        
        # The backward function expects a gradient for each input it received.
        # The 'dim' input does not require a gradient, so we return None for it.
        return x_float.grad, None

# ==============================================================================
# FINAL, CORRECTED NN.MODULE WRAPPER
# ==============================================================================

class IntegerApproximatedSoftmax(nn.Module):
    """
    A module wrapper for the integer-approximated softmax function.
    
    Args:
        dim (int): The dimension along which softmax will be computed.
    """
    def __init__(self, dim=-1):
        super().__init__()
        # CORRECT: Store the dimension as a member variable.
        self.dim = dim

    def forward(self, x):
        # CORRECT: Pass the dimension to our custom autograd function.
        return IntegerApproximatedSoftmaxFn.apply(x, self.dim)
    

# ==============================================================================