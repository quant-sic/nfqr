from typing import List, Union

import torch
from numpy import pi
from pydantic import BaseModel
from torch.nn import Module, ModuleList

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.layers.conditioners import ConditionerChain
from nfqr.normalizing_flows.layers.layer_config import LAYER_REGISTRY, LayerConfig
from nfqr.normalizing_flows.layers.layer_splits import SPLIT_TYPES_REGISTRY
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class LayerChainConfig(BaseModel):

    dim: List[int]
    layer_configs: Union[None, LayerConfig, List[LayerConfig]]
    connect_splits: bool = True


class LayerChain(Module):
    def __init__(
        self,
        dim: List[int],
        layer_configs: Union[None, LayerConfig, List[LayerConfig]],
        connect_splits: bool = True,
        **kwargs,
    ):

        super(LayerChain, self).__init__()

        self.layers = ModuleList()
        self.dim = dim

        if layer_configs is None:
            raise ValueError("Layer Configs None not handled")
        elif not isinstance(layer_configs, list):
            layer_configs = [layer_configs]

        split_num_offsets = [0] + list(map(lambda c: c.num_layers, layer_configs))
        for layer_config_idx, layer_config in enumerate(layer_configs):

            layer_splits = SPLIT_TYPES_REGISTRY[
                layer_config.layer_split_config.split_type
            ](
                num_layers=layer_config.num_layers,
                num_offset=split_num_offsets[layer_config_idx] if connect_splits else 0,
                dim=dim,
                **dict(
                    layer_config.layer_split_config.specific_split_type_config
                    if layer_config.layer_split_config.specific_split_type_config
                    else {}
                ),
            )

            if layer_config.layer_type not in ("coupling_layer",) and (
                layer_config.conditioner_chain_config.share_encoder
                or layer_config.conditioner_chain_config.share_decoder
            ):
                raise ValueError(
                    "conditioner sharing only implemented for Coupling layers"
                )

            conditioners = ConditionerChain(
                **dict(layer_config.conditioner_chain_config),
                layer_splits=layer_splits,
                num_pars=DIFFEOMORPHISMS_REGISTRY[
                    layer_config.specific_layer_config.domain
                ][layer_config.specific_layer_config.diffeomorphism_config.diffeomorphism_type](**(dict(layer_config.specific_layer_config.diffeomorphism_config.specific_diffeomorphism_config if layer_config.specific_layer_config.diffeomorphism_config.specific_diffeomorphism_config is not None else {}))).num_pars,
            )

            for (conditioner_mask, transformed_mask), conditioner in zip(
                layer_splits, conditioners
            ):
                if layer_config.layer_type in ("ar_layer",):
                    if connect_splits:
                        logger.warning(
                            "Notice that an Autoregressive layer breaks the split connection!"
                        )

                c = LAYER_REGISTRY._registry[layer_config.layer_type][
                    layer_config.specific_layer_config.specific_layer_type
                ](
                    conditioner_mask=conditioner_mask,
                    transformed_mask=transformed_mask,
                    **dict(layer_config.specific_layer_config),
                    dim=self.dim,
                    conditioner=conditioner,
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
