import torch
from numpy import pi
from torch.nn import functional as F

from nfqr.normalizing_flows.diffeomorphisms.diffeomorphism_base import Diffeomorphism
from nfqr.normalizing_flows.diffeomorphisms.inversion import (
    NumericalInverse,
    searchsorted,
)
from nfqr.normalizing_flows.misc.constraints import (
    greater_than_eq,
    nf_constraints_standard,
    simplex,
    torch_transform_to,
)
from nfqr.registry import StrRegistry

U1_DIFFEOMORPHISM_REGISTRY = StrRegistry("u1")


def bring_back_to_u1(phi):
    if (torch.min(phi) < 0.0) or (torch.max(phi) > (2 * pi)):
        if torch.min(phi) > -(1e-3) and torch.max(phi) < (2 * pi + 1e-3):
            phi[phi < 0.0] = 0.0
            phi[phi > (2 * pi)] = 2 * pi
        else:
            raise ValueError(
                f"Min:{torch.min(phi)}, Max:{torch.max(phi)} outside of domain"
            )

    return phi


def ncp(phi, alpha, beta, rho, ret_logabsdet=True):

    left_bound_mask = phi < 1e-3
    right_bound_mask = phi > (2 * pi - 1e-3)

    out = 2 * torch.atan(alpha * torch.tan(0.5 * (phi - pi))[..., None] + beta) + pi
    out[left_bound_mask] = phi[left_bound_mask][..., None] / alpha[left_bound_mask]
    out[right_bound_mask] = (
        2 * pi + (phi[right_bound_mask][..., None] - 2 * pi) / alpha[right_bound_mask]
    )

    conv_comb = (rho * out).sum(-1)

    if ret_logabsdet:
        _grad = (
            (1 + beta**2) * torch.sin(phi / 2).pow(2)[..., None] / alpha
            + alpha * torch.cos(phi / 2).pow(2)[..., None]
            - beta * torch.sin(phi)[..., None]
        ).pow(-1)

        _grad[left_bound_mask | right_bound_mask] = (
            1 / alpha[left_bound_mask | right_bound_mask]
        )

        logabsdet = torch.log((rho * _grad).sum(-1))

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


@U1_DIFFEOMORPHISM_REGISTRY.register("ncp")
class NCP(Diffeomorphism):
    def __init__(self, alpha_min=1e-3) -> None:
        super(NCP).__init__()

        self._num_pars = 3

        self.inverse_fn_params = {
            "function": ncp,
            "args": ["alpha", "beta", "rho"],
            "left": 0.0,
            "right": 2 * pi,
            "kwargs": {"ret_logabsdet": False},
        }

        self.alpha_transform = nf_constraints_standard(greater_than_eq(alpha_min))
        self.rho_transform = torch_transform_to(simplex)

    @property
    def num_pars(self):
        return self._num_pars

    def constrain_params(self, alpha_unconstrained, beta, rho_unconstrained):

        alpha = self.alpha_transform(alpha_unconstrained)
        rho = self.rho_transform(rho_unconstrained)

        return alpha, beta, rho

    def __call__(
        self, phi, alpha_unconstrained, beta, rho_unconstrained, ret_logabsdet=True
    ):

        alpha, beta, rho = self.constrain_params(
            alpha_unconstrained, beta, rho_unconstrained
        )

        phi = bring_back_to_u1(phi)

        if ret_logabsdet:

            phi_out,ld = ncp(
                        phi=phi, alpha=alpha, beta=beta, rho=rho, ret_logabsdet=ret_logabsdet
                    )
            phi_out = bring_back_to_u1(phi_out)

            return phi_out,ld
        
        else:

            phi_out = ncp(
                        phi=phi, alpha=alpha, beta=beta, rho=rho, ret_logabsdet=ret_logabsdet
                    )
            phi_out = bring_back_to_u1(phi_out)

            return phi_out

    def inverse(
        self, phi, alpha_unconstrained, beta, rho_unconstrained, ret_logabsdet=True
    ):
        alpha, beta, rho = self.constrain_params(
            alpha_unconstrained, beta, rho_unconstrained
        )

        phi = bring_back_to_u1(phi)

        phi_out = NumericalInverse.apply(phi, self.inverse_fn_params, alpha, beta, rho)
        phi_out = bring_back_to_u1(phi_out)

        if ret_logabsdet:
            _, ld = ncp(
                phi=phi,
                alpha=alpha,
                beta=beta,
                rho=rho,
                ret_logabsdet=True,
            )

            return phi_out, -ld
        else:
            return phi_out


def moebius(phi, w, rho, ret_logabsdet=True):

    # dims (phi) = batch_size x dim(input)
    # dims (w) = 2 x dims(phi) x #convex_comb
    # dims (rho) = dims(phi) x #convex_comb

    # phi to complex number
    # stack with ones, st. transformation on one ist performed
    z = torch.stack(
        [
            torch.stack([torch.cos(phi), torch.sin(phi)]),
            torch.stack(
                [
                    torch.ones(*phi.shape, device=phi.device),
                    torch.zeros(*phi.shape, device=phi.device),
                ]
            ),
        ]
    )

    # dims(z) = 2(phi and ones) x 2(complex) x batch_size x values_per_dimension x convex_combinations

    z_min_w = z[..., None] - w[None, ...]
    z_min_w_squared_norm = (z_min_w**2).sum(dim=1)
    one_min_w_squared = 1 - (w**2).sum(dim=0)

    # factor in formula for moebius
    beta = one_min_w_squared[None, :] / z_min_w_squared_norm

    # resulting complex number
    h_w = beta[:, None, ...] * z_min_w - w[None, ...]

    angles = torch.atan2(h_w[:, 1, ...], h_w[:, 0, ...])
    # apply rotation such that moebius(0)=0
    angles = (angles[0] - angles[1]) % (2 * pi)

    convex_comb = (rho * angles).sum(dim=-1)
    convex_comb = convex_comb % (2 * pi)

    # calc log_absdet
    if ret_logabsdet:

        alpha = 2 * beta[0] / z_min_w_squared_norm[0]
        z_min_w_xy = z_min_w[0, 0] * z_min_w[0, 1] * alpha
        z_min_w_squared = (z_min_w[0] ** 2) * alpha

        d = (
            h_w[0, 1] * (beta[0] - z_min_w_squared[0]) + h_w[0, 0] * z_min_w_xy
        ) * torch.sin(phi)[..., None] + (
            h_w[0, 1] * z_min_w_xy + h_w[0, 0] * (beta[0] - z_min_w_squared[1])
        ) * torch.cos(
            phi
        )[
            ..., None
        ]

        logabsdet = torch.log((rho * d).sum(-1))

        return convex_comb, logabsdet

    else:
        return convex_comb


@U1_DIFFEOMORPHISM_REGISTRY.register("moebius")
class Moebius(Diffeomorphism):
    def __init__(self) -> None:
        super(Moebius).__init__()

        self._num_pars = 3

        self.inverse_fn_params = {
            "function": moebius,
            "args": ["w", "rho"],
            "left": 0.0,
            "right": 2 * pi,
            "kwargs": {"ret_logabsdet": False},
        }

        self.rho_transform = torch_transform_to(simplex)

    @property
    def num_pars(self):
        return self._num_pars

    def constrain_params(self, w_x_unconstrained, w_y_unconstrained, rho_unconstrained):

        # normalize rho
        rho = self.rho_transform(rho_unconstrained)
        w_unconstrained = torch.stack((w_x_unconstrained, w_y_unconstrained), dim=0)
        # get w into circle ? sigmoid a good option ? restrain angle 0-2pi ?

        w_unconstrained = torch.norm(
            torch.stack((w_x_unconstrained, w_y_unconstrained), dim=0), p=2, dim=0
        )

        w = (
            0.99
            * w_unconstrained
            / (1 + torch.norm(w_unconstrained, p=2, dim=0))[None, ...]
        )

        # torch.sigmoid(w_mag_unbounded) * torch.stack(
        #     [torch.cos(w_angle_unbounded),
        #      torch.sin(w_angle_unbounded)], dim=0)

        return w, rho

    def __call__(
        self,
        phi,
        w_x_unconstrained,
        w_y_unconstrained,
        rho_unconstrained,
        ret_logabsdet=True,
    ):

        w, rho = self.constrain_params(
            w_x_unconstrained, w_y_unconstrained, rho_unconstrained
        )

        phi = bring_back_to_u1(phi)

        return moebius(
            phi=phi,
            w=w,
            rho=rho,
            ret_logabsdet=ret_logabsdet,
        )

    def inverse(
        self,
        phi,
        w_x_unconstrained,
        w_y_unconstrained,
        rho_unconstrained,
        ret_logabsdet=True,
    ):
        w, rho = self.constrain_params(
            w_x_unconstrained, w_y_unconstrained, rho_unconstrained
        )

        phi = bring_back_to_u1(phi)
        phi_out = NumericalInverse.apply(phi, self.inverse_fn_params, w, rho)

        if ret_logabsdet:
            _, ld = moebius(
                phi=phi,
                w=w,
                rho=rho,
                ret_logabsdet=True,
            )

            return phi_out, -ld
        else:
            return phi_out


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse,
    left,
    right,
    bottom,
    top,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    bounds_error=1e-2,
    ret_logabsdet=True,
    circular=True,
):
    """
    Rational quadratic spline transform
    """

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    if circular:
        derivatives = F.pad(derivatives, pad=(0, 1), mode="constant", value=0.0)
        derivatives[..., -1] = derivatives[..., 0]
    else:
        if not derivatives.shape[-1] == widths.shape[-1] + 1:
            raise ValueError(
                f"Invalid dim of derivatives !(K+1) {derivatives.shape} for non-circular case"
            )

    assert (derivatives > 0.0).all()

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives[..., :-1].gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        if ret_logabsdet:
            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator)

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        if ret_logabsdet:
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    if ret_logabsdet:
        return outputs, logabsdet
    else:
        return outputs


@U1_DIFFEOMORPHISM_REGISTRY.register("rqs")
class RQS(Diffeomorphism):
    def __init__(self) -> None:
        super(RQS).__init__()

        self._num_pars = 3

    @property
    def num_pars(self):
        return self._num_pars

    def constrain_params(
        self, unnormalized_widths, unnormalized_heights, unnormalized_derivatives
    ):
        pass

    def __call__(
        self,
        phi,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        ret_logabsdet=True,
    ):
        phi = bring_back_to_u1(phi)
        return rational_quadratic_spline(
            inputs=phi,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=False,
            left=0.0,
            right=2 * pi,
            bottom=0.0,
            top=2 * pi,
            ret_logabsdet=ret_logabsdet,
        )

    def inverse(
        self,
        phi,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        ret_logabsdet=True,
    ):
        phi = bring_back_to_u1(phi)
        return rational_quadratic_spline(
            inputs=phi,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=True,
            left=0.0,
            right=2 * pi,
            bottom=0.0,
            top=2 * pi,
            ret_logabsdet=ret_logabsdet,
        )
