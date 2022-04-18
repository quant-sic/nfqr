import torch
from torch.nn import Module, parameter

from nfqr.normalizing_flows.coupling_layers.conditioner import ConditionerU1
from nfqr.normalizing_flows.diffeomorphisms.u1.ncp import NCPInverse, ncp
from nfqr.normalizing_flows.misc.constraints import (
    greater_than_eq,
    nf_constraints_standard,
    simplex,
    torch_transform_to,
)


class Coupling(Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
    ) -> None:
        super(Coupling, self).__init__()

        self.conditioner_mask = conditioner_mask
        self.transformed_mask = transformed_mask

    def _split(self, xz):

        return xz[..., self.conditioner_mask], xz[..., self.transformed_mask]

    def decode(self):
        pass

    def encode(self):
        pass


class NCPCoupling(Coupling):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        net,
        expressivity,
        alpha_min=1e-3,
        **kwargs
    ):
        super(NCPCoupling, self).__init__(conditioner_mask, transformed_mask)
        self.conditioner = ConditionerU1(
            net,
            conditioner_mask.sum().item(),
            transformed_mask.sum().item(),
            expressivity=expressivity,
            num_splits=3,
            **kwargs
        )

        # self.conditioner_shift = ConditionerU1(
        #     net,
        #     conditioner_mask.sum().item(),
        #     transformed_mask.sum().item(),
        #     expressivity=1,
        #     num_splits=1,
        #     **kwargs
        # )

        self.ncp_inverse = NCPInverse.apply
        self.alpha_transform = nf_constraints_standard(greater_than_eq(alpha_min))
        self.rho_transform = torch_transform_to(simplex)

    def get_conditioner_params(self, conditioner_input, transformed_input):

        alpha_unbounded, beta, rho_unnormalized = self.conditioner(conditioner_input)
        # shift = self.conditioner_shift(conditioner_input)[0].reshape(
        #     transformed_input.shape
        # )

        alpha = self.alpha_transform(alpha_unbounded)
        rho = self.rho_transform(rho_unnormalized)

        return alpha, beta, rho

    def decode(self, z):

        conditioner_input, transformed_input = self._split(z)
        alpha, beta, rho = self.get_conditioner_params(
            conditioner_input=conditioner_input, transformed_input=transformed_input
        )

        z[..., self.transformed_mask], ld = ncp(
            phi=transformed_input,
            alpha=alpha,
            beta=beta,
            rho=rho,
            ret_logabsdet=True,
        )

        log_det = ld.sum(dim=-1)

        return z, log_det

    def encode(self, x):

        conditioner_input, transformed_input = self._split(x)
        alpha, beta, rho = self.get_conditioner_params(
            conditioner_input=conditioner_input, transformed_input=transformed_input
        )

        x[..., self.transformed_mask] = self.ncp_inverse(
            transformed_input, alpha, beta, rho
        )
        _, ld = ncp(
            phi=x[..., self.transformed_mask],
            alpha=alpha,
            beta=beta,
            rho=rho,
            ret_logabsdet=True,
        )

        # - bc for inverse logabsdet
        log_det = -ld.sum(dim=-1)

        return x, log_det


class IdentityCoupling(Coupling):
    def __init__(self, conditioner_mask, transformed_mask, **kwargs):
        super(IdentityCoupling, self).__init__(conditioner_mask, transformed_mask)

    def decode(self, z):

        log_det = torch.zeros(z.shape[0])

        return z, log_det

    def encode(self, x):

        log_det = torch.zeros(x.shape[0])

        return x, log_det


class ResCoupling(Coupling):
    def __init__(self, coupling, **kwargs):
        super(ResCoupling, self).__init__(
            coupling.conditioner_mask, coupling.transformed_mask
        )

        self.coupling = coupling
        self.identity = IdentityCoupling(
            coupling.conditioner_mask, coupling.transformed_mask
        )

        self.rho_unnormalized = parameter.Parameter(
            torch.full(size=(2,), fill_value=0.5)
        )
        self.rho_transform = nf_constraints_standard(simplex)

    def decode(self, z):

        z_coupling, log_det_coupling = self.coupling.decode(z.clone())
        z_identity, log_det_identity = self.identity.decode(z.clone())

        log_rho = self.rho_transform(self.rho_unnormalized)

        z[..., self.transformed_mask] = (
            log_rho[0].exp() * z_coupling[..., self.transformed_mask]
            + log_rho[1].exp() * z_identity[..., self.transformed_mask]
        )

        log_det = torch.logsumexp(
            torch.stack(
                [
                    log_rho[0] + log_det_coupling,
                    log_rho[1] + log_det_identity,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return z, log_det

    def encode(self, x):

        x_coupling, log_det_coupling = self.coupling.encode(x.clone())
        x_identity, log_det_identity = self.identity.encode(x.clone())

        log_rho = self.rho_transform(self.rho_unnormalized)

        x[..., self.transformed_mask] = (
            log_rho[0].exp() * x_coupling[..., self.transformed_mask]
            + log_rho[1].exp() * x_identity[..., self.transformed_mask]
        )

        log_det = torch.logsumexp(
            torch.stack(
                [
                    log_rho[0] + log_det_coupling,
                    log_rho[1] + log_det_identity,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return x, log_det
