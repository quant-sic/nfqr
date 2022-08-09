from functools import cached_property
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.nn import Module

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.layers.conditioners import (
    ConditionerChain,
    ConditionerChainConfig,
)
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

logger = create_logger(__name__)

AR_LAYER_REGISTRY = StrRegistry("ar_layer")
N_TRANSFORMS_LAYER_REGISTRY = StrRegistry("n_transforms_layer")


class LayerBase(Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        conditioner_chain_config: ConditionerChainConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super(LayerBase, self).__init__()

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][diffeomorphism]()
        self.conditioner_chain_config = conditioner_chain_config
        self.domain = domain
        self.dim = dim

    def decode(self, z):
        pass

    def encode(self, x):
        pass


@AR_LAYER_REGISTRY.register("iterative")
class IterativeARLayer(LayerBase, Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        conditioner_chain_config: ConditionerChainConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super().__init__(
            dim, diffeomorphism, conditioner_chain_config, domain, **kwargs
        )

        if not len(dim) == 1:
            raise ValueError("Layer not yet constructed for multidimensional input dim")

        self.conditioners = torch.nn.ModuleList(
            list(
                ConditionerChain(
                    **dict(conditioner_chain_config),
                    layer_splits=zip(self.conditioner_masks, self.transformed_masks),
                    num_pars=self.diffeomorphism.num_pars,
                )
            )[1:]
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
    # net_config: NetConfig

    # validators ..


@N_TRANSFORMS_LAYER_REGISTRY.register("single")
class SingleTransformLayer(LayerBase, Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        conditioner_chain_config: ConditionerChainConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super().__init__(
            dim, diffeomorphism, conditioner_chain_config, domain, **kwargs
        )

        if not len(dim) == 1:
            raise ValueError("Layer not yet constructed for multidimensional input dim")

        self.conditioners = torch.nn.ModuleList(
            list(
                ConditionerChain(
                    **dict(conditioner_chain_config),
                    layer_splits=zip(self.conditioner_masks, self.transformed_masks),
                    num_pars=self.diffeomorphism.num_pars,
                )
            )
        )

    @staticmethod
    def single_transform_mask(size, mask_num, n_transformed=1, **kwargs):

        mask = torch.zeros(size).bool()
        step_size = int(np.ceil(size / n_transformed))

        transformed_idx = (torch.arange(n_transformed) * step_size + mask_num) % size
        mask[transformed_idx] = True

        return ~mask, mask

    @cached_property
    def conditioner_masks(self):
        return [
            self.single_transform_mask(self.dim[0], idx)[0]
            for idx in range(self.dim[0])
        ]

    @cached_property
    def transformed_masks(self):
        return [
            self.single_transform_mask(self.dim[0], idx)[1]
            for idx in range(self.dim[0])
        ]

    def decode(self, z):

        x = z.clone()
        log_det = torch.zeros(z.shape[0], device=z.device)

        for idx in range(z.shape[-1]):

            unconstrained_params = self.conditioners[idx](
                z[..., self.conditioner_masks[idx]]
            )
            x[..., self.transformed_masks[idx]], ld = self.diffeomorphism(
                z[..., self.transformed_masks[idx]],
                *unconstrained_params,
                ret_logabsdet=True,
            )
            log_det += ld.squeeze()

        return x, log_det

    # impossible like this since conditioner input gone !!!!
    def encode(self, x):

        z = x.clone()
        log_det = torch.zeros(x.shape[0], device=x.device)

        for idx in range(x.shape[-1]):

            unconstrained_params = self.conditioners[idx](
                x[..., self.conditioner_masks[idx]]
            )
            z[..., self.transformed_masks[idx]], ld = self.diffeomorphism.inverse(
                z[..., self.transformed_masks[idx]],
                *unconstrained_params,
                ret_logabsdet=True,
            )

            log_det += ld.squeeze()

        return z, log_det


class NTransformsLayerConfig(BaseModel):

    domain: Literal["u1"] = "u1"
    specific_layer_type: N_TRANSFORMS_LAYER_REGISTRY.enum = Field(...)
    diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum
    expressivity: int
    # net_config:

    # validators ..
