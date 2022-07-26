from typing import List, Union

import torch
from numpy import pi
from pydantic import BaseModel
from torch.nn import Module, ModuleList

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.layers import conditioners
from nfqr.normalizing_flows.layers.conditioners import CONDITIONER_REGISTRY
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

        self.layer_chain_conditioners = torch.nn.ModuleList()

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

            if layer_config.share_conditioner:
                if not layer_splits.all_conditioners_equal_in_out:
                    raise RuntimeError(
                        "share conditioners not possible since in and output dimensions do not match"
                    )
                if layer_config.layer_type not in ("coupling_layer",):
                    raise ValueError(
                        "conditioner sharing only implemented for Coupling layers"
                    )

                dim_in_0, dim_out_0 = map(
                    lambda m: m.sum().item(), layer_splits.__iter__().__next__()
                )

                shared_conditioner = CONDITIONER_REGISTRY[
                    layer_config.specific_layer_config.domain
                ](
                    dim_in=dim_in_0,
                    dim_out=dim_out_0,
                    expressivity=layer_config.specific_layer_config.expressivity,
                    num_splits=DIFFEOMORPHISMS_REGISTRY[
                        layer_config.specific_layer_config.domain
                    ][layer_config.specific_layer_config.diffeomorphism]().num_pars,
                    net_config=layer_config.specific_layer_config.net_config,
                )
                self.layer_chain_conditioners.append(shared_conditioner)
            else:
                shared_conditioner = None

            for (conditioner_mask, transformed_mask), _ in zip(
                layer_splits, range(layer_config.num_layers)
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
                    conditioner=shared_conditioner,
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
