from typing import Literal

import torch
from torch.nn import Module, parameter

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.misc.constraints import nf_constraints_standard, simplex
from nfqr.normalizing_flows.nets.config import NetConfig
from nfqr.registry import StrRegistry

from .conditioners import CONDITIONER_REGISTRY

COUPLING_TYPES = StrRegistry("coupling_types")


class CouplingLayer(Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs
    ) -> None:
        super(CouplingLayer, self).__init__()

        self.conditioner_mask = conditioner_mask
        self.transformed_mask = transformed_mask

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][diffeomorphism]()

        self.conditioner = CONDITIONER_REGISTRY[domain](
            dim_in=conditioner_mask.sum().item(),
            dim_out=transformed_mask.sum().item(),
            expressivity=expressivity,
            num_splits=self.diffeomorphism.num_pars,
            net_config=net_config,
        )

    def _split(self, xz):
        return xz[..., self.conditioner_mask], xz[..., self.transformed_mask]

    def decode(self, z):
        pass

    def encode(self, x):
        pass


@COUPLING_TYPES.register("bare")
class BareCoupling(CouplingLayer):
    def decode(self, z):

        conditioner_input, transformed_input = self._split(z)

        unconstrained_params = self.conditioner(conditioner_input)

        z[..., self.transformed_mask], ld = self.diffeomorphism(
            transformed_input, *unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return z, log_det

    def encode(self, x):

        conditioner_input, transformed_input = self._split(x)

        unconstrained_params = self.conditioner(conditioner_input)

        x[..., transformed_input], ld = self.diffeomorphism.inverse(
            transformed_input, *unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return x, log_det


@COUPLING_TYPES.register("residual")
class ResidualCoupling(CouplingLayer, Module):
    def __init__(self, conditioner_mask, transformed_mask, config, **kwargs):
        super(ResidualCoupling, self).__init__(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            config=config,
        )

        self.rho_unnormalized = parameter.Parameter(
            torch.full(size=(2,), fill_value=0.5)
        )
        self.rho_transform = nf_constraints_standard(simplex)

    def decode(self, z):

        conditioner_input, transformed_input = self._split(z)
        unconstrained_params = self.conditioner(conditioner_input)

        z_coupling, log_det_coupling = self.diffeomorphism(
            transformed_input.clone(), *unconstrained_params, ret_logabsdet=True
        )

        log_rho = self.rho_transform(self.rho_unnormalized)

        z[..., self.transformed_mask] = (
            log_rho[0].exp() * z_coupling
            + log_rho[1].exp() * z.clone()[..., self.transformed_mask]
        )

        log_det = torch.logsumexp(
            torch.stack(
                [
                    log_rho[0] + log_det_coupling,
                    log_rho[1] * torch.ones_like(log_det_coupling),
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return z, log_det

    def encode(self, x):

        conditioner_input, transformed_input = self._split(x)
        unconstrained_params = self.conditioner(conditioner_input)

        x_coupling, log_det_coupling = self.self.diffeomorphism(
            transformed_input.clone(), *unconstrained_params, ret_logabsdet=True
        )

        log_rho = self.rho_transform(self.rho_unnormalized)

        x[..., self.transformed_mask] = (
            log_rho[0].exp() * x_coupling[..., self.transformed_mask]
            + log_rho[1].exp() * x.clone()[..., self.transformed_mask]
        )

        log_det = torch.logsumexp(
            torch.stack(
                [
                    log_rho[0] + log_det_coupling,
                    log_rho[1],
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return x, log_det
