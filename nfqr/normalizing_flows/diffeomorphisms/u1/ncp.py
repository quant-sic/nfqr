from functools import partial

import torch
from numpy import pi
from torch.autograd import Function, grad

from nfqr.normalizing_flows.diffeomorphisms.inversion import bisection_invert


def ncp(phi, alpha, beta, rho, ret_logabsdet=True):

    if (torch.min(phi) < 0.0) or (torch.max(phi) > (2 * pi)):
        if torch.min(phi) > -(1e-2) and torch.max(phi) < (2 * pi + 1e-2):
            phi[phi < 0.0] = 0.0
            phi[phi > (2 * pi)] = 2 * pi
        else:
            raise ValueError(
                f"Min:{torch.min(phi)}, Max:{torch.max(phi)} outside of domain"
            )

    left_bound_mask = phi < 1e-3
    right_bound_mask = phi > (2 * pi - 1e-3)

    out = 2 * torch.atan(alpha * torch.tan(0.5 * (phi - pi))[..., None] + beta) + pi
    out[left_bound_mask] = phi[left_bound_mask][..., None] / alpha[left_bound_mask]
    out[right_bound_mask] = (
        2 * pi + (phi[right_bound_mask][..., None] - 2 * pi) / alpha[right_bound_mask]
    )

    conv_comb = (rho * out).sum(-1)

    if ret_logabsdet:
        grad = (
            (1 + beta**2) * torch.sin(phi / 2).pow(2)[..., None] / alpha
            + alpha * torch.cos(phi / 2).pow(2)[..., None]
            - beta * torch.sin(phi)[..., None]
        ).pow(-1)

        grad[left_bound_mask | right_bound_mask] = (
            1 / alpha[left_bound_mask | right_bound_mask]
        )

        logabsdet = torch.log((rho * grad).sum(-1))

        return conv_comb, logabsdet
    else:
        return conv_comb


# def ncp_mod(phi, alpha_unbound, beta, rho_unnormalized, ret_logabsdet=True):

#     alpha = F.softplus(alpha_unbound) + 1e-3  # exponential function
#     rho = torch.softmax(rho_unnormalized, dim=-1)

#     # left_bound_mask = phi < 1e-3
#     # right_bound_mask = phi > (2*pi-1e-3)

#     out = 2 * torch.atan(alpha * torch.tan(0.5 * (phi - pi))[..., None] + beta) + pi
#     # out[left_bound_mask] = phi[left_bound_mask][..., None]/alpha[left_bound_mask]
#     # out[right_bound_mask] = 2*pi + \
#     #     (phi[right_bound_mask][..., None]-2*pi)/alpha[right_bound_mask]

#     conv_comb = (rho * out).sum(-1)
#     conv_comb = conv_comb % (2 * pi)

#     if ret_logabsdet:
#         grad = (
#             (1 + beta**2) * torch.sin(phi / 2).pow(2)[..., None] / alpha
#             + alpha * torch.cos(phi / 2).pow(2)[..., None]
#             - beta * torch.sin(phi)[..., None]
#         ).pow(-1)

#         # grad[left_bound_mask | right_bound_mask] = 1 / \
#         #     alpha[left_bound_mask | right_bound_mask]

#         logabsdet = torch.log((rho * grad).sum(-1))

#         return conv_comb, logabsdet
#     else:
#         return conv_comb


class NCPInverse(Function):
    @staticmethod
    def forward(ctx, x, par_1, par_2, par_3):

        func_partial = partial(
            ncp,
            alpha=par_1,
            beta=par_2,
            rho=par_3,
            ret_logabsdet=False,
        )
        z = bisection_invert(func_partial, x, 0.0, 2 * pi, tol=1e-4)

        ctx.save_for_backward(z, par_1, par_2, par_3)

        return z

    @staticmethod
    def backward(ctx, grad_z):

        with torch.enable_grad():

            z, par_1, par_2, par_3 = map(
                lambda t: t.detach().clone().requires_grad_(), ctx.saved_tensors
            )

            x = ncp(z, par_1, par_2, par_3, ret_logabsdet=False)

            (grad_x_inverse,) = grad(x, z, torch.ones_like(x), retain_graph=True)

            grad_x = grad_z * grad_x_inverse.pow(-1)
            grad_par_1, grad_par_2, grad_par_3 = grad(x, [par_1, par_2, par_3], -grad_x)

        return grad_x, grad_par_1, grad_par_2, grad_par_3


# class NCPModInverse(Function):
#     @staticmethod
#     def forward(ctx, x, par_1, par_2, par_3):

#         func_partial = partial(
#             ncp_mod,
#             alpha=par_1,
#             beta=par_2,
#             rho=par_3,
#             ret_logabsdet=False,
#         )
#         z = bisection_invert(func_partial, x, 0.0, 2 * pi, tol=1e-4)

#         ctx.save_for_backward(z, par_1, par_2, par_3)

#         return z

#     @staticmethod
#     def backward(ctx, grad_z):

#         with torch.enable_grad():

#             z, par_1, par_2, par_3 = map(
#                 lambda t: t.detach().clone().requires_grad_(), ctx.saved_tensors
#             )

#             x = ncp_mod(z, par_1, par_2, par_3, ret_logabsdet=False)

#             (grad_x_inverse,) = grad(x, z, torch.ones_like(x), retain_graph=True)

#             grad_x = grad_z * grad_x_inverse.pow(-1)
#             grad_par_1, grad_par_2, grad_par_3 = grad(x, [par_1, par_2, par_3], -grad_x)

#         return grad_x, grad_par_1, grad_par_2, grad_par_3
