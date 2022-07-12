from typing import Dict, List, Union

import torch
from numpy import pi
from pydantic import BaseModel
from torch.nn import Module, ModuleList

from nfqr.normalizing_flows.layers.autoregressive_layers import (
    AR_LAYER_TYPES,
    ARLayerConfig,
)
from nfqr.normalizing_flows.layers.coupling_layers import (
    COUPLING_TYPES,
    SPLIT_TYPES,
    CouplingConfig,
    generate_splits,
)
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class LayerChainConfig(BaseModel):

    dim: List[int]
    layers_config: Union[
        None, CouplingConfig, ARLayerConfig, List[Union[ARLayerConfig, CouplingConfig]]
    ]
    split_type: Union[SPLIT_TYPES, None]
    num_layers: int


class LayerChain(Module):
    def __init__(
        self,
        dim: List[int],
        layers_config: List[Dict],
        split_type: SPLIT_TYPES,
        num_layers: int,
        **kwargs,
    ):

        super(LayerChain, self).__init__()

        self.layers = ModuleList()
        self.dim = dim

        if split_type in (SPLIT_TYPES.single_transforms,):
            if len(dim) > 1:
                raise ValueError("n dim >1 not implemented for single_transforms splitting")
            elif dim[0] != num_layers:
                logger.info(
                    f"Single transforms splitting will result in num_layers == dim({dim})"
                )
                num_layers = dim[0]
                if isinstance(layers_config, list) and len(layers_config) != dim[0]:
                    raise ValueError(
                        f"Single transforms splitting needs {dim[0]} layers but {len(layers_config)} were given"
                    )

        splits = generate_splits(split_type, num_layers, dim)

        if layers_config is None:
            return

        if not isinstance(layers_config, list):
            layers_config = [layers_config] * num_layers

        for (conditioner_mask, transformed_mask), layer_config in zip(
            splits, layers_config
        ):
            if isinstance(layer_config, CouplingConfig):
                c = COUPLING_TYPES[layer_config.coupling_type](
                    conditioner_mask=conditioner_mask,
                    transformed_mask=transformed_mask,
                    **dict(layer_config),
                )
            elif isinstance(layer_config, ARLayerConfig):
                c = AR_LAYER_TYPES[layer_config.ar_layer_type](
                    dim=self.dim, **dict(layer_config)
                )
            else:
                raise NotImplementedError()

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
