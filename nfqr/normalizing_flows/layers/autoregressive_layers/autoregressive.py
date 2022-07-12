from functools import cached_property
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch.nn import Module

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.layers.conditioners import CONDITIONER_REGISTRY
from nfqr.normalizing_flows.nets import NetConfig
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

logger = create_logger(__name__)

AR_LAYER_REGISTRY = StrRegistry("ar_layer")


class AutoregressiveLayer(Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super(AutoregressiveLayer, self).__init__()

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][diffeomorphism]()
        self.expressivity = expressivity
        self.net_config = net_config
        self.domain = domain
        self.dim = dim

    def decode(self, z):
        pass

    def encode(self, x):
        pass


@AR_LAYER_REGISTRY.register("iterative")
class IterativeARLayer(AutoregressiveLayer, Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super().__init__(
            dim, diffeomorphism, expressivity, net_config, domain, **kwargs
        )

        if not len(dim) == 1:
            raise ValueError("Layer not yet constructed for multidimensional input dim")

        self.conditioners = torch.nn.ModuleList()
        for idx in range(1, dim[0]):
            self.conditioners.append(
                CONDITIONER_REGISTRY[domain](
                    dim_in=idx,
                    dim_out=1,
                    expressivity=expressivity,
                    num_splits=self.diffeomorphism.num_pars,
                    net_config=net_config,
                )
            )

    @staticmethod
    def autoregressive_mask(size, idx):

        # rewrite, this just works in 1d
        mask_conditioner = torch.ones(size).cumsum(-1) <= idx

        mask_transformed = torch.zeros(size).bool()
        mask_transformed[idx] = True

        return mask_conditioner, mask_transformed

    @cached_property
    def conditioner_masks(self):
        return [
            self.autoregressive_mask(self.dim[0], idx)[0] for idx in range(self.dim[0])
        ]

    @cached_property
    def transformed_masks(self):
        return [
            self.autoregressive_mask(self.dim[0], idx)[1] for idx in range(self.dim[0])
        ]

    def decode(self, z):

        x = z.clone()
        log_det = torch.zeros(z.shape[0], device=z.device)

        for idx in range(1, z.shape[-1]):

            unconstrained_params = self.conditioners[idx - 1](
                z[..., self.conditioner_masks[idx]]
            )
            x[..., self.transformed_masks[idx]], ld = self.diffeomorphism(
                z[..., self.transformed_masks[idx]],
                *unconstrained_params,
                ret_logabsdet=True,
            )
            log_det += ld.squeeze()

        return x, log_det

    def encode(self, x):

        z = x.clone()
        log_det = torch.zeros(x.shape[0], device=x.device)

        for idx in range(1, x.shape[-1]):

            unconstrained_params = self.conditioners[idx - 1](
                x[..., self.conditioner_masks[idx]]
            )
            z[..., self.transformed_masks[idx]], ld = self.diffeomorphism.inverse(
                z[..., self.transformed_masks[idx]],
                *unconstrained_params,
                ret_logabsdet=True,
            )

            log_det += ld.squeeze()

        return z, log_det


class ARLayerConfig(BaseModel):

    domain: Literal["u1"] = "u1"
    specific_layer_type: AR_LAYER_REGISTRY.enum = Field(...)
    diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum
    expressivity: int
    net_config: NetConfig

    # validators ..
