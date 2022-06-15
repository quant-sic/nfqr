from typing import Dict, List, Tuple, Union

import torch
from numpy import pi
from pydantic import BaseModel
from torch.nn import Module, ModuleList

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
    layers_config: Union[None, CouplingConfig, List[CouplingConfig]]
    split_type: SPLIT_TYPES
    num_layers: int


class LayerChain(Module):
    def __init__(
        self,
        dim: List[int],
        layers_config: List[Dict],
        split_type: SPLIT_TYPES,
        num_layers: int,
        **kwargs
    ):

        super(LayerChain, self).__init__()

        self.layers = ModuleList()
        self.size = dim

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
                    **dict(layer_config)
                )
            else:
                raise NotImplementedError()

            self.layers.append(c)

    def encode(self, x):

        assert (x >= 0.0).all() and (x <= (2 * pi)).all()

        # x.shape[0] extracts batch dim
        abs_log_det = torch.zeros(x.shape[0], device=x.device)

        for coupling in self.layers[::-1]:
            x, ld = coupling.encode(x)
            abs_log_det += ld

        return x, abs_log_det

    def decode(self, z):

        if not ((z >= 0.0) & (z <= (2 * pi))).all():
            logger.info(z)

        log_det = torch.zeros(z.shape[0], device=z.device)

        for coupling in self.layers:
            z, ld = coupling.decode(z)
            log_det += ld

        return z, log_det

    def load(self, checkpoint, device):
        self.load_state_dict(torch.load(checkpoint, map_location=device)["net"])
