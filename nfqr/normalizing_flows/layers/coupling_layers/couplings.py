from functools import cached_property
from math import pi
from typing import Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.nn import Module, parameter

from nfqr.normalizing_flows.diffeomorphisms import (
    DIFFEOMORPHISMS_REGISTRY,
    DiffeomorphismConfig,
)
from nfqr.normalizing_flows.diffeomorphisms.inversion import NumericalInverse
from nfqr.normalizing_flows.layers.conditioners import ConditionerChain
from nfqr.normalizing_flows.misc.constraints import nf_constraints_standard, simplex
from nfqr.normalizing_flows.nets.decoder import DecoderConfig, MLPDecoderConfig
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

logger = create_logger(__name__)

COUPLING_LAYER_REGISTRY = StrRegistry("coupling_layer")


class CouplingLayer(Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        conditioner=None,
        **kwargs,
    ) -> None:
        super(CouplingLayer, self).__init__()

        self.conditioner_mask = conditioner_mask
        self.transformed_mask = transformed_mask

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][
            diffeomorphism_config.diffeomorphism_type
        ](
            **(
                dict(
                    diffeomorphism_config.specific_diffeomorphism_config
                    if diffeomorphism_config.specific_diffeomorphism_config is not None
                    else {}
                )
            )
        )

        if conditioner is not None:
            self.conditioner = conditioner

    # def conditioner(self):
    #     pass

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


@COUPLING_LAYER_REGISTRY.register("translation_equivariant")
class TranslationEquivariantCoupling(CouplingLayer):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        conditioner=None,
        **kwargs,
    ) -> None:

        super().__init__(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism_config=diffeomorphism_config,
            domain=domain,
            conditioner=conditioner,
            **kwargs,
        )

        self.check_correct_splitting()

    def check_correct_splitting(self):
        assert not (
            torch.roll(self.transformed_mask, shifts=-1, dims=0) & self.conditioner_mask
        ).any(), "Masks are not suitable for equivariant coupling"

    def _split_diffs_equivariant(self, z):

        diffs = self.diffeomorphism.diff_to_range(z - torch.roll(z, shifts=1, dims=-1))
        diffs_for_conditioner, diffs_to_be_transformed = (
            diffs[..., self.conditioner_mask],
            diffs[..., self.transformed_mask],
        )

        return diffs_for_conditioner, diffs_to_be_transformed

    def _decode(self, z):

        diffs_for_conditioner, diffs_to_be_transformed = self._split_diffs_equivariant(
            z
        )

        unconstrained_params = self.conditioner(diffs_for_conditioner)

        transformed_input, ld = self.diffeomorphism(
            diffs_to_be_transformed, unconstrained_params, ret_logabsdet=True
        )

        delta = transformed_input - diffs_to_be_transformed

        z[..., self.transformed_mask] = self.diffeomorphism.map_to_range(
            z[..., self.transformed_mask] + delta
        )
        log_det = ld.sum(dim=-1)

        return z, log_det, diffs_to_be_transformed, transformed_input

    def _encode(self, x):

        diffs_for_conditioner, diffs_to_be_transformed = self._split_diffs_equivariant(
            x
        )
        unconstrained_params = self.conditioner(diffs_for_conditioner)

        transformed_input, ld = self.diffeomorphism.inverse(
            diffs_to_be_transformed, unconstrained_params, ret_logabsdet=True
        )

        delta = transformed_input - diffs_to_be_transformed
        x[..., self.transformed_mask] = self.diffeomorphism.map_to_range(
            x[..., self.transformed_mask] - delta
        )
        log_det = ld.sum(dim=-1)

        return x, log_det


@COUPLING_LAYER_REGISTRY.register("bare")
class BareCoupling(CouplingLayer):
    def _decode(self, z):

        conditioner_input, transformed_input = self._split(z)

        unconstrained_params = self.conditioner(conditioner_input)

        z[..., self.transformed_mask], ld = self.diffeomorphism(
            transformed_input, unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return z, log_det

    def _encode(self, x):

        conditioner_input, transformed_input = self._split(x)

        unconstrained_params = self.conditioner(conditioner_input)

        x[..., self.transformed_mask], ld = self.diffeomorphism.inverse(
            transformed_input, unconstrained_params, ret_logabsdet=True
        )

        log_det = ld.sum(dim=-1)

        return x, log_det


class ResidualCoupling(CouplingLayer, Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        residual_type="global",
        initial_rho_id=None,
        conditioner=None,
        **kwargs,
    ) -> None:
        super().__init__(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism_config=diffeomorphism_config,
            domain=domain,
            conditioner=conditioner,
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

            if initial_rho_id is None:
                raise ValueError(
                    "intial_rho_id cannot be set to None for global residual"
                )

            self.rho_unnormalized = parameter.Parameter(
                torch.zeros(2, dtype=torch.float32), requires_grad=requires_grad
            )
            self.rho_unnormalized[self.rho_assignment["diff"]] = self.rho_unnormalized[
                self.rho_assignment["id"]
            ] + np.log((1 - initial_rho_id) / initial_rho_id)

            self.get_log_rho = self.get_log_rho_global

        elif residual_type == "conditioned":
            if conditioner_mask.sum().item() > 0:
                self.rho_net = (
                    ConditionerChain(
                        encoder_config=None,
                        decoder_config=DecoderConfig(
                            decoder_type="mlp",
                            specific_decoder_config=MLPDecoderConfig(
                                net_hidden=[
                                    conditioner_mask.sum().item(),
                                    int(conditioner_mask.sum().item() + 1 / 2),
                                ]
                            ),
                        ),
                        domain=domain,
                        layer_splits=((conditioner_mask, transformed_mask),),
                        expressivity=2,
                        num_pars=1,
                        share_encoder=False,
                        share_decoder=False,
                    )
                    .__iter__()
                    .__next__()
                )

            self.get_log_rho = self.get_log_rho_conditioned

        else:
            raise ValueError(f"Unknown Residual type {residual_type}")

    @classmethod
    def as_global_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        initial_rho_id: float = 0.5,
        conditioner=None,
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism_config=diffeomorphism_config,
            domain=domain,
            residual_type="global",
            initial_rho_id=initial_rho_id,
            conditioner=conditioner,
        )

    @classmethod
    def as_global_non_trainable_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        initial_rho_id: float = 0.5,
        conditioner=None,
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism_config=diffeomorphism_config,
            domain=domain,
            residual_type="global_non_trainable",
            initial_rho_id=initial_rho_id,
            conditioner=conditioner,
        )

    @classmethod
    def as_conditioned_residual(
        cls,
        conditioner_mask,
        transformed_mask,
        diffeomorphism_config: DiffeomorphismConfig,
        domain: Literal["u1"] = "u1",
        conditioner=None,
        **kwargs,
    ):
        return cls(
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
            diffeomorphism_config=diffeomorphism_config,
            domain=domain,
            residual_type="conditioned",
            conditioner=conditioner,
        )

    @cached_property
    def rho_assignment(self):
        return {"diff": 0, "id": 1}

    def convex_comb(self, z, log_rho, unconstrained_params, ret_logabsdet=True):

        if ret_logabsdet:
            z_coupling, log_det_coupling = self.diffeomorphism(
                z.clone(), unconstrained_params, ret_logabsdet=ret_logabsdet
            )
        else:
            z_coupling = self.diffeomorphism(
                z.clone(), unconstrained_params, ret_logabsdet=ret_logabsdet
            )

        z = self.diffeomorphism.range_check_correction(
            log_rho[..., self.rho_assignment["diff"]].exp() * z_coupling
            + log_rho[..., self.rho_assignment["id"]].exp() * z.clone()
        )

        if ret_logabsdet:
            ld = torch.logsumexp(
                torch.stack(
                    [
                        log_rho[..., self.rho_assignment["diff"]] + log_det_coupling,
                        log_rho[..., self.rho_assignment["id"]]
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
            return {
                "rho": {
                    "id": rho[self.rho_assignment["id"]],
                    "diff": rho[self.rho_assignment["diff"]],
                }
            }
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
        z_out = self.diffeomorphism.range_check_correction(x)

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
    diffeomorphism_config: DiffeomorphismConfig
    initial_rho_id: Optional[float]
    # validators ..
