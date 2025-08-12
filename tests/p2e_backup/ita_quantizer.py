#
# FILE: ita_quantizer.py
#
import itertools
from typing import Any, Dict, Optional, Type
import os, sys

import torch.nn as nn

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

# Core torchao imports
from torchao.quantization.pt2e.quantizer import Quantizer, QuantizationAnnotation
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    is_valid_annotation,
)

from torchao.quantization.pt2e.quantizer import (
    QuantizationConfig,
    DerivedQuantizationSpec
)

from torchao.testing.pt2e._xnnpack_quantizer_utils import _is_annotated

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tuki.ita_quantization_specs import (
    _derive_bias_qparams_fn,
)

# Our custom components
from models.ITA.ITA_layers import ITASoftmax
from ita_quantization_specs import QuantizationConfig

class ITAQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.module_type_config: Dict[Type[nn.Module], Optional[QuantizationConfig]] = {}
        self.softmax_config: Optional[QuantizationConfig] = None
        self.op_configs: Dict[Any, QuantizationConfig] = {}

    def set_module_type(self, module_type: Type[nn.Module], config: QuantizationConfig):
        self.module_type_config[module_type] = config
        return self

    def set_softmax_qconfig(self, config: QuantizationConfig):
        self.softmax_config = config
        return self
    
    def set_config_for(self, op_type: Any, config: QuantizationConfig):
        """Sets a quantization config for a specific operator or module type."""
        self.op_configs[op_type] = config
        return self

    def annotate(self, model: GraphModule) -> GraphModule:
        """
        Annotates the graph by finding all instances of configured operator/module
        types and applying their respective quantization configs.
        """
        # A map to link ATen targets back to their parent module type for config lookup
        aten_op_to_module_map = {
            torch.ops.aten.convolution.default: nn.Conv2d,
            torch.ops.aten.addmm.default: nn.Linear,
            torch.ops.aten.bmm.default: torch.matmul,
            torch.ops.aten.matmul.default: torch.matmul,
            torch.ops.aten.lstm.input: nn.LSTM,
            torch.ops.aten.softmax: ITASoftmax,
        }

        # Find all partitions for all configured ops at once
        all_op_types = list(self.op_configs.keys())
        all_partitions = get_source_partitions(model.graph, all_op_types)

        for op_type, partitions in all_partitions.items():
            config = self.op_configs.get(op_type)
            if not config:
                continue

            for part in partitions:
                # Dispatch to the correct annotation function based on the partition's
                # primary operator node.
                output_node = part.output_nodes[0]
                if output_node.target == torch.ops.aten.convolution.default:
                    self._annotate_conv2d(output_node, config)
                elif output_node.target == torch.ops.aten.addmm.default:
                    self._annotate_linear(output_node, config)
                elif output_node.target in [torch.ops.aten.bmm.default, torch.ops.aten.matmul.default]:
                    self._annotate_matmul(output_node, config)
                elif output_node.target == torch.ops.aten.lstm.input:
                    self._annotate_lstm(output_node, config)
                elif output_node.target == torch.ops.aten.softmax.default:
                    self._annotate_softmax(output_node, config)

        return model

    def _annotate_conv2d(self, conv_node: Node, config: QuantizationConfig):
        # Annotation logic for a single conv node
        input_act, weight, bias = conv_node.args[0], conv_node.args[1], conv_node.args[2]
        input_qspec_map = {
            input_act: config.input_activation,
            weight: config.weight,
        }
        if isinstance(bias, Node):
            # For standard ARM config, bias is float
            input_qspec_map[bias] = config.bias
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=config.output_activation,
        )

    def _annotate_linear(self, addmm_node: Node, config: QuantizationConfig):
        # Annotation logic for a single linear node
        bias, input_act, weight = addmm_node.args
        input_qspec_map = {
            input_act: config.input_activation,
            weight: config.weight,
        }
        if isinstance(bias, Node):
            # If this is the accelerator config, create the derived spec.
            # Otherwise, use the standard float bias from the ARM config.
            if config.bias and config.bias.dtype == torch.float32: # This is the ARM config
                 input_qspec_map[bias] = config.bias
            else: # This is the accelerator config
                bias_qspec = DerivedQuantizationSpec(
                    derived_from=[(input_act, addmm_node), (weight, addmm_node)],
                    derive_qparams_fn=_derive_bias_qparams_fn,
                    dtype=torch.int32, quant_min=-(2**31), quant_max=2**31 - 1,
                    qscheme=torch.per_channel_symmetric,
                )
                input_qspec_map[bias] = bias_qspec
        addmm_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=config.output_activation,
        )

    def _annotate_matmul(self, matmul_node: Node, config: QuantizationConfig):
        # Annotation logic for a single matmul node
        input_a, input_b = matmul_node.args[0], matmul_node.args[1]
        matmul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_a: config.input_activation,
                input_b: config.input_activation,
            }, output_qspec=config.output_activation,
        )

    def _annotate_softmax(self, softmax_node: Node, config: QuantizationConfig):
        # Annotation logic for a single softmax node
        input_act = softmax_node.args[0]
        softmax_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={input_act: config.input_activation},
            output_qspec=config.output_activation,
        )

    def _annotate_lstm(self, lstm_node: Node, config: QuantizationConfig):
        """Annotates the aten.lstm.input operator."""
        # args: input, hx, params, has_biases, num_layers, ...
        input_act, _, params, has_biases, _, _, _, _, _, _ = lstm_node.args
        input_qspec_map = {input_act: config.input_activation}
        
        # LSTM weights and biases are packed into the `params` list
        if has_biases: # weight_ih, weight_hh, bias_ih, bias_hh for each layer
             for i in range(0, len(params), 4):
                 input_qspec_map[params[i]] = config.weight # weight_ih
                 input_qspec_map[params[i+1]] = config.weight # weight_hh
                 input_qspec_map[params[i+2]] = config.bias # bias_ih
                 input_qspec_map[params[i+3]] = config.bias # bias_hh
        else: # weight_ih, weight_hh for each layer
            for i in range(0, len(params), 2):
                 input_qspec_map[params[i]] = config.weight # weight_ih
                 input_qspec_map[params[i+1]] = config.weight # weight_hh
        
        lstm_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            # LSTM output is a tuple (output, h, c). We quantize the main output.
            output_qspec=config.output_activation,
        )
        
    def validate(self, model: GraphModule) -> None:
        """
        A method to validate the annotated graph. For now, we will just pass.
        """
        pass
