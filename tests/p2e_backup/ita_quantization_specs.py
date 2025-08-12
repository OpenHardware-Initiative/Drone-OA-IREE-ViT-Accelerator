#
# FILE: ita_quantization_specs.py
#
from typing import List, Tuple
import torch
from torchao.quantization.pt2e.observer import (
    ObserverBase,
    HistogramObserver,
    PerChannelMinMaxObserver,
    MinMaxObserver,
    PlaceholderObserver,
)
# Note: QuantizationSpec and its variants are now imported from torchao
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpec,
    QuantizationConfig,
    DerivedQuantizationSpec
)

# This function is the key to handling int32 accumulators for bias.
# It computes the bias scale as: scale_bias = scale_input * scale_weight
def _derive_bias_qparams_fn(
    obs_or_fqs: List[ObserverBase],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derives the quantization parameters for the bias from the activation and weight
    observers. This ensures the quantized bias is correctly scaled for an
    int32 accumulator.
    """
    if len(obs_or_fqs) != 2:
        raise ValueError(f"Expected 2 observers (activation, weight), got {len(obs_or_fqs)}")
    
    act_obs, weight_obs = obs_or_fqs[0], obs_or_fqs[1]
    
    # Calculate qparams for activation and weight
    act_scale, _ = act_obs.calculate_qparams()
    weight_scale, _ = weight_obs.calculate_qparams()
    
    # The bias scale is the product of the input activation and weight scales
    bias_scale = act_scale * weight_scale
    
    # Bias zero point is 0 for symmetric quantization of weights
    bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
    
    return bias_scale, bias_zero_point

def get_arm_symmetric_qconfig() -> QuantizationConfig:
    """
    Returns a standard symmetric quantization config suitable for ARM CPUs.
    - Bias is kept as float32 and handled by the backend.
    """
    act_spec = QuantizationSpec(
        dtype=torch.int8, quant_min=-128, quant_max=127,
        qscheme=torch.per_tensor_symmetric, observer_or_fake_quant_ctr=HistogramObserver
    )
    weight_spec = QuantizationSpec(
        dtype=torch.int8, quant_min=-127, quant_max=127,
        qscheme=torch.per_channel_symmetric, ch_axis=0, observer_or_fake_quant_ctr=PerChannelMinMaxObserver
    )
    # Standard ARM backends expect the bias to be fp32.
    bias_spec = QuantizationSpec(
        dtype=torch.float32, observer_or_fake_quant_ctr=PlaceholderObserver
    )
    return QuantizationConfig(
        input_activation=act_spec, output_activation=act_spec,
        weight=weight_spec, bias=bias_spec,
    )

def get_ita_accelerator_qconfig() -> QuantizationConfig:
    """
    Configuration for standard operations (MatMul, Linear) on your accelerator.
    - Activations (Inputs): int8, per-tensor, symmetric.
    - Weights: int8, per-channel, symmetric.
    - Bias: int32, with scale derived from input and weight scales.
    """
    act_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=HistogramObserver
    )

    weight_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver
    )
    
    bias_spec_placeholder = QuantizationSpec(
        dtype=torch.float32,  # The bias is initially float before being quantized
        observer_or_fake_quant_ctr=PlaceholderObserver
    )
    
    # Use DerivedQuantizationSpec for the bias
    #bias_spec = DerivedQuantizationSpec(
    #    # This spec will derive its qparams from the activation and weight
    #    # which will be specified during annotation
    #    derive_qparams_fn=_derive_bias_qparams_fn,
    #    dtype=torch.int32,
    #    quant_min=-(2**31),
    #    quant_max=2**31 - 1,
    #    qscheme=torch.per_tensor_symmetric, # Bias is per-channel
    #)

    return QuantizationConfig(
        input_activation=act_spec,
        output_activation=act_spec,
        weight=weight_spec,
        bias=bias_spec_placeholder,
        # Set derived_from during annotation, not here
    )

def get_ita_softmax_qconfig() -> QuantizationConfig:
    """
    Special configuration for the Softmax operation.
    - Input: int8, symmetric.
    - Output: uint8, asymmetric (affine).
    """
    input_act_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=HistogramObserver,
    )

    output_act_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=255,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver,
    )

    return QuantizationConfig(
        input_activation=input_act_spec,
        output_activation=output_act_spec,
        weight=None,
        bias=None,
    )