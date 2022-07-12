from typing import Dict, List, Union

import torch
from numpy import pi
from pydantic import BaseModel
from torch.nn import Module, ModuleList

from nfqr.normalizing_flows.layers.autoregressive_layers import (
    AR_LAYER_REGISTRY,
    ARLayerConfig,
)
from nfqr.normalizing_flows.layers.coupling_layers import (
    COUPLING_LAYER_REGISTRY,
    CouplingConfig,
)
from nfqr.normalizing_flows.layers.layer_config import LAYER_REGISTRY, LayerConfig
from nfqr.normalizing_flows.layers.layer_splits import (
    SPLIT_TYPES_REGISTRY,
    LayerSplitConfig,
)
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class LayerChainConfig(BaseModel):

    dim: List[int]
    layer_configs: Union[None, LayerConfig, List[LayerConfig]]
    layer_split_config: Union[LayerSplitConfig, None]
    num_layers: int


class LayerChain(Module):
    def __init__(
        self,
        dim: List[int],
        layer_configs: List[Dict],
        layer_split_config: Union[LayerSplitConfig, None],
        num_layers: int,
        **kwargs,
    ):

        super(LayerChain, self).__init__()

        self.layers = ModuleList()
        self.dim = dim
        layer_splits = SPLIT_TYPES_REGISTRY[layer_split_config.split_type](
            num_layers=num_layers,
            dim=dim,
            **dict(
                layer_split_config.specific_split_type_config
                if layer_split_config.specific_split_type_config
                else {}
            ),
        )

        if layer_configs is None:
            raise ValueError("Layer Configs None not handled")

        if not isinstance(layer_configs, list):
            layer_configs = [layer_configs] * num_layers

        for (conditioner_mask, transformed_mask), layer_config in zip(
            layer_splits, layer_configs
        ):
            c = LAYER_REGISTRY._registry[layer_config.layer_type][
                layer_config.specific_layer_config.specific_layer_type
            ](
                conditioner_mask=conditioner_mask,
                transformed_mask=transformed_mask,
                **dict(layer_config.specific_layer_config),
                dim=self.dim,
            )

            self.layers.append(c)

    def encode(self, x):

        assert (x >= 0.0).all() and (x <= (2 * pi)).all()

        # x.shape[0] extracts batch dim
        abs_log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers[::-1]:
            x, ld = layer.encode(x)
            abs_log_det += ld

        return x, abs_log_det

    def decode(self, z):

        if not ((z >= 0.0) & (z <= (2 * pi))).all():
            logger.info(z)

        log_det = torch.zeros(z.shape[0], device=z.device)

        for layer in self.layers:
            z, ld = layer.decode(z)
            log_det += ld

        return z, log_det

    def load(self, checkpoint, device):
        self.load_state_dict(torch.load(checkpoint, map_location=device)["net"])
