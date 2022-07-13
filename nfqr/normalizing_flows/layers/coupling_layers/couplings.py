from functools import cached_property
from math import pi
from typing import Literal

import torch
from pydantic import BaseModel, Field
from torch.nn import Module, parameter

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.diffeomorphisms.inversion import NumericalInverse
from nfqr.normalizing_flows.layers.conditioners import CONDITIONER_REGISTRY
from nfqr.normalizing_flows.misc.constraints import nf_constraints_standard, simplex
from nfqr.normalizing_flows.nets import NetConfig
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

logger = create_logger(__name__)

COUPLING_LAYER_REGISTRY = StrRegistry("coupling_layer")


class CouplingLayer(Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super(CouplingLayer, self).__init__()

        self.conditioner_mask = conditioner_mask
        self.transformed_mask = transformed_mask

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][diffeomorphism]()

        if conditioner_mask.sum().item() > 0:
            self.conditioner = CONDITIONER_REGISTRY[domain](
                dim_in=conditioner_mask.sum().item(),
                dim_out=transformed_mask.sum().item(),
                expressivity=expressivity,
                num_splits=self.diffeomorphism.num_pars,
                net_config=net_config,
            )

    def _split(self, xz):
        return xz[..., self.conditioner_mask], xz[..., self.transformed_mask]

    def _decode(self, z):
        pass

    def _encode(self, z):
        pass

    def decode(self, z):

        if self.conditioner_mask.sum().item() == 0:
            return z, torch.zeros(z.shape[0], device=z.device)
        else:
            return self._decode(z=z)

    def encode(self, x):

        if self.conditioner_mask.sum().item() == 0:
            return x, torch.zeros(x.shape[0], device=x.device)
        else:
            return self._encode(x=x)


@COUPLING_LAYER_REGISTRY.register("bare")
class BareCoupling(CouplingLayer):
    def _decode(self, z):

        conditioner_input, transformed_input = self._split(z)

        unconstrained_params = self.conditioner(conditioner_input)

        z[..., self.transformed_mask], ld = self.diffeomorphism(
            transformed_input, *unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return z, log_det

    def _encode(self, x):

        conditioner_input, transformed_input = self._split(x)

        unconstrained_params = self.conditioner(conditioner_input)

        x[..., self.transformed_mask], ld = self.diffeomorphism.inverse(
            transformed_input, *unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return x, log_det


class ResidualCoupling(CouplingLayer, Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        residual_type="global",
        **kwargs,
    ) -> None:
        super().__init__(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism=diffeomorphism,
            expressivity=expressivity,
            net_config=net_config,
            domain=domain,
            **kwargs,
        )

        self.residual_type = residual_type

        # not ideal since params could be constrained before bisection search
        self.inverse_fn_params = {
            "function": self.convex_comb,
            "args": ["log_rho", "unconstrained_params"],
            "left": 0.0,
            "right": 2 * pi,
            "kwargs": {"ret_logabsdet": False},
        }

        if "global" in residual_type:
            if "non_trainable" in residual_type:
                requires_grad = False
            else:
                requires_grad = True

            self.rho_unnormalized = parameter.Parameter(
                torch.full(size=(2,), fill_value=0.5), requires_grad=requires_grad
            )
            self.get_log_rho = self.get_log_rho_global

        elif residual_type == "conditioned":
            if conditioner_mask.sum().item() > 0:
                self.rho_net = CONDITIONER_REGISTRY[domain](
                    dim_in=conditioner_mask.sum().item(),
                    dim_out=transformed_mask.sum().item(),
                    expressivity=2,
                    num_splits=1,
                    net_config=NetConfig(
                        net_type="mlp",
                        net_hidden=[
                            conditioner_mask.sum().item(),
                            int(conditioner_mask.sum().item() / 2),
                        ],
                    ),
                )
            self.get_log_rho = self.get_log_rho_conditioned

        else:
            raise ValueError(f"Unknown Residual type {residual_type}")

    @classmethod
    def as_global_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism=diffeomorphism,
            expressivity=expressivity,
            net_config=net_config,
            domain=domain,
            residual_type="global",
        )

    @classmethod
    def as_global_non_trainable_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism=diffeomorphism,
            expressivity=expressivity,
            net_config=net_config,
            domain=domain,
            residual_type="global_non_trainable",
        )

    @classmethod
    def as_conditioned_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism=diffeomorphism,
            expressivity=expressivity,
            net_config=net_config,
            domain=domain,
            residual_type="conditioned",
        )

    def convex_comb(self, z, log_rho, unconstrained_params, ret_logabsdet=True):

        if ret_logabsdet:
            z_coupling, log_det_coupling = self.diffeomorphism(
                z.clone(), *unconstrained_params, ret_logabsdet=ret_logabsdet
            )
        else:
            z_coupling = self.diffeomorphism(
                z.clone(), *unconstrained_params, ret_logabsdet=ret_logabsdet
            )

        z = self.diffeomorphism.map_to_range(
            log_rho[..., 0].exp() * z_coupling + log_rho[..., 1].exp() * z.clone()
        )

        if ret_logabsdet:
            ld = torch.logsumexp(
                torch.stack(
                    [
                        log_rho[..., 0] + log_det_coupling,
                        log_rho[..., 1]
                        + torch.zeros_like(
                            log_det_coupling, device=log_det_coupling.device
                        ),
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
            return z, ld
        else:
            return z

    @cached_property
    def rho_transform(self):
        return nf_constraints_standard(simplex)

    def get_log_rho_conditioned(self, conditioner_input):
        (log_rho_unconstrained,) = self.rho_net(conditioner_input)
        log_rho = self.rho_transform(log_rho_unconstrained)

        if not torch.allclose(
            log_rho.exp().sum(dim=-1), torch.ones(log_rho.shape[:-1])
        ):
            logger.info(log_rho.exp().sum(dim=-1).max())
            logger.info(log_rho.exp().sum(dim=-1).min())

        return log_rho

    def get_log_rho_global(self, *args, **kwargs):
        return self.rho_transform(self.rho_unnormalized)

    @property
    def logging_parameters(self):
        if hasattr(self, "rho_unnormalized"):
            rho = self.rho_transform(self.rho_unnormalized).exp().clone().detach()
            return {"rho": {"id": rho[1], "diff": rho[0]}}
        else:
            return {}

    def _decode(self, z):

        conditioner_input, transformed_input = self._split(z)
        unconstrained_params = self.conditioner(conditioner_input)
        log_rho = self.get_log_rho(conditioner_input=conditioner_input)

        z[..., self.transformed_mask], ld = self.convex_comb(
            log_rho=log_rho,
            z=transformed_input,
            unconstrained_params=unconstrained_params,
            ret_logabsdet=True,
        )

        log_det = ld.sum(dim=-1)

        return z, log_det

    def _encode(self, x):

        conditioner_input, transformed_input = self._split(x)
        unconstrained_params = torch.stack(self.conditioner(conditioner_input), dim=0)
        log_rho = self.get_log_rho(conditioner_input=conditioner_input)

        x[..., self.transformed_mask] = NumericalInverse.apply(
            transformed_input.clone(),
            self.inverse_fn_params,
            log_rho,
            unconstrained_params,
        )
        z_out = self.diffeomorphism.map_to_range(x)

        _, ld = self.convex_comb(
            z=z_out[..., self.transformed_mask],
            log_rho=log_rho,
            unconstrained_params=unconstrained_params,
            ret_logabsdet=True,
        )

        log_det = ld.sum(dim=-1)

        return z_out, -log_det


COUPLING_LAYER_REGISTRY.register("global_residual", ResidualCoupling.as_global_residual)
COUPLING_LAYER_REGISTRY.register(
    "global_non_trainable_residual", ResidualCoupling.as_global_non_trainable_residual
)
COUPLING_LAYER_REGISTRY.register(
    "conditioned_residual", ResidualCoupling.as_conditioned_residual
)


class CouplingConfig(BaseModel):

    domain: Literal["u1"] = "u1"
    specific_layer_type: COUPLING_LAYER_REGISTRY.enum = Field(...)
    diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum
    expressivity: int
    net_config: NetConfig

    # validators ..
