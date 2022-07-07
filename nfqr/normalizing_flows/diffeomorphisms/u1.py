import torch
from numpy import pi
from torch.nn import functional as F
import numpy as np
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


def bring_back_to_u1(phi,**kwargs):
    if (torch.min(phi) < 0.0) or (torch.max(phi) > (2 * pi)):
        if torch.min(phi) > -(1e-2) and torch.max(phi) < (2 * pi + 1e-2):
            phi[phi < 0.0] = 0.0
            phi[phi > (2 * pi)] = 2 * pi
        else:
            kwargs_str = ";".join([f"{k}:min {torch.min(v)} max {torch.max(v)}" for k,v in kwargs.items()])
            raise ValueError(
                f"Min:{torch.min(phi)}, Max:{torch.max(phi)} outside of domain. {kwargs_str}"
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


def ncp_mod(phi, alpha, beta, rho, ret_logabsdet=True):


    out = 2 * torch.atan(alpha * torch.tan(0.5 * (phi - pi))[..., None] + beta) + pi

    conv_comb = (rho * out).sum(-1)
    conv_comb = conv_comb % (2 * pi)

    if ret_logabsdet:
        grad = (
            (1 + beta**2) * torch.sin(phi / 2).pow(2)[..., None] / alpha
            + alpha * torch.cos(phi / 2).pow(2)[..., None]
            - beta * torch.sin(phi)[..., None]
        ).pow(-1)

        logabsdet = torch.log((rho * grad).sum(-1))

        return conv_comb, logabsdet
    else:
        return conv_comb


@U1_DIFFEOMORPHISM_REGISTRY.register("ncp")
class NCP(Diffeomorphism):
    def __init__(self, alpha_min=1e-3,boundary_mode="taylor") -> None:
        super(NCP).__init__()

        self._num_pars = 3

        if boundary_mode == "taylor":
            self.fn = ncp
        elif boundary_mode == "modulo":
            self.fn = ncp_mod
        else:
            raise ValueError(f"Unknown Boundary mode {boundary_mode}")

        self.inverse_fn_params = {
            "function": self.fn,
            "args": ["alpha", "beta", "rho"],
            "left": 0.0,
            "right": 2 * pi,
            "kwargs": {"ret_logabsdet": False},
        }

        self.alpha_transform = nf_constraints_standard(greater_than_eq(alpha_min))
        self.rho_transform = torch_transform_to(simplex)

    @classmethod
    def use_modulo_for_boundary(cls,alpha_min=1e-3):
        return cls(alpha_min,boundary_mode="modulo")

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

            phi_out,ld = self.fn(
                        phi=phi, alpha=alpha, beta=beta, rho=rho, ret_logabsdet=ret_logabsdet
                    )
            phi_out = bring_back_to_u1(phi_out)

            return phi_out,ld
        
        else:

            phi_out = self.fn(
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
            _, ld = self.fn(
                phi=phi_out,
                alpha=alpha,
                beta=beta,
                rho=rho,
                ret_logabsdet=True,
            )

            return phi_out, -ld
        else:
            return phi_out

U1_DIFFEOMORPHISM_REGISTRY.register("ncp_mod",NCP.use_modulo_for_boundary)



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

        d = (h_w[0, 1] * (beta[0] - z_min_w_squared[0]) + h_w[0, 0] * z_min_w_xy) * torch.sin(phi)[..., None] + (h_w[0, 1] * z_min_w_xy + h_w[0, 0] * (beta[0] - z_min_w_squared[1])) * torch.cos(phi)[..., None]

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

        # w_unconstrained = torch.norm(
        #     torch.stack((w_x_unconstrained, w_y_unconstrained), dim=0), p=2, dim=0
        # )

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


        if ret_logabsdet:

            phi_out,ld = moebius(
                            phi=phi,
                            w=w,
                            rho=rho,
                            ret_logabsdet=ret_logabsdet,
                        )
            phi_out = bring_back_to_u1(phi_out)

            return phi_out,ld
        
        else:

            phi_out = moebius(
                        phi=phi,
                        w=w,
                        rho=rho,
                        ret_logabsdet=ret_logabsdet,
                    )
            phi_out = bring_back_to_u1(phi_out)

            return phi_out



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
        phi_out = bring_back_to_u1(phi_out)

        if ret_logabsdet:
            _, ld = moebius(
                phi=phi_out,
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


    outputs = bring_back_to_u1(outputs)

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

def rho_step_function(x, alpha, beta):

    if isinstance(x, (torch.Tensor, np.ndarray)):
        out = torch.zeros_like(x)
        out[x != 0] = torch.exp(-1 / (alpha * x**beta))[x != 0]
        dout = torch.zeros_like(x)
        dout[x != 0] = (torch.exp(-1 / (alpha * x**beta)) *
                        (beta / (alpha * x**(beta + 1))))[x != 0]

    elif isinstance(x, (float, int)):
        out = 0 if x == 0 else np.exp(-1 / (alpha * x**beta))
        dout = 0 if x == 0 else np.exp(
            -1 / (alpha * x**beta)) * (beta / (alpha * x**(beta + 1)))

    return out, dout


def generalized_sigmoid(x, rho_function, alpha, beta):

    rho_x, _ = rho_function(x, alpha, beta)
    rho_1_x, _ = rho_function(1 - x, alpha, beta)
    return rho_x / (rho_x + rho_1_x)


def dg(phi, rho_function, a, b, phi_left, phi_right, alpha, beta):

    dg_x = torch.zeros_like(phi)

    inside_mask = (phi < phi_right) & (phi > phi_left)

    # masking needs to happen before any nan producing operations are done. Otherwise nans will propagate to grad calculation !!!!!
    y_inside = (a * (phi - b) + 0.5)[inside_mask]
    alpha_inside = alpha[inside_mask]
    beta_inside = beta[inside_mask]
    a_inside = a[inside_mask]

    rho_y, drho_y = rho_function(y_inside, alpha_inside, beta_inside)
    rho_1_y, drho_1_y = rho_function(1 - y_inside, alpha_inside, beta_inside)
    divisor = rho_y + rho_1_y

    dg_x[inside_mask] = ((drho_y - (rho_y * (drho_y - drho_1_y) / divisor)) * a_inside / divisor)

    return dg_x


def g(phi, rho_function, a, b, phi_left, phi_right, alpha, beta):

    g_x = torch.zeros_like(phi)
    g_x[phi >= phi_right] = 1.0

    inside_mask = (phi < phi_right) & (phi > phi_left)

    y = a * (phi - b) + 0.5
    g_x[inside_mask] = generalized_sigmoid(y[inside_mask], rho_function,
                                           alpha[inside_mask],
                                           beta[inside_mask])

    return g_x


def circular_bump(phi,
                  rho,
                  a,
                  b,
                  c,
                  alpha,
                  beta,
                  rho_function=rho_step_function,
                  ret_logabsdet=True):

    phi = phi[...,None].expand(*rho.shape)/(2*pi)

    phi_left = b - 1 / (2 * a)
    phi_right = b + 1 / (2 * a)

    phi_shifted = phi.clone()
    phi_shifted[(phi_right > 1) & (phi <= phi_right - 1)] += 1
    phi_shifted[(phi_left < 0) & (phi >= 1 + phi_left)] -= 1

    # calculate g
    g_phi = g(phi_shifted, rho_function, a, b, phi_left, phi_right, alpha,
              beta)

    #cumulative bump
    left_value = torch.zeros_like(phi)
    left_value[~((phi_left < 0) | (phi_right < 1))] = 1.0
    left = g(left_value, rho_function, a, b, phi_left, phi_right, alpha, beta)
    in_between_mask = ((phi_right > 1) &
                       (phi >
                        (phi_right % 1))) | ((phi_left < 0) & (phi >
                                                               (phi_left % 1)))

    left[in_between_mask] = -1.0 + left[in_between_mask]
    f_phi = ((g_phi - left) * (1 - c) + c * phi)

    # convex sum
    f_phi_out = (rho * f_phi).sum(dim=-1) * 2 * pi

    if ret_logabsdet:

        # calculate dg
        dg_phi = dg(phi_shifted, rho_function, a, b, phi_left, phi_right,
                    alpha, beta) * (1 - c) + c

        dg_phi_out = torch.log((rho * dg_phi).sum(dim=-1))

        return f_phi_out, dg_phi_out

    else:
        return f_phi_out



@U1_DIFFEOMORPHISM_REGISTRY.register("bump")
class Bump(Diffeomorphism):
    def __init__(self,_beta:int=2) -> None:
        super(Bump).__init__()

        self._num_pars = 5

        self.inverse_fn_params = {
            "function": circular_bump,
            "args": ["rho", "a","b","c","alpha","beta"],
            "left": 0.0,
            "right": 2 * pi,
            "kwargs": {"ret_logabsdet": False},
        }
        
        self.alpha_max = 10
        self._beta = _beta
        self.rho_transform = torch_transform_to(simplex)

    @property
    def num_pars(self):
        return self._num_pars

    def constrain_params(self, 
                        rho_unconstrained,
                        a_unconstrained,
                        b_unconstrained,
                        c_unconstrained,
                        alpha_unconstrained
                        ):

        rho = torch.softmax(rho_unconstrained, dim=-1)

        a = F.softplus(a_unconstrained) + 1 + 1e-3
        b = b_unconstrained**2 / (1 + b_unconstrained**2)
        c = c_unconstrained**2 / (1 + c_unconstrained**2)

        alpha = (F.tanh(alpha_unconstrained)+1)* self.alpha_max + 1e-3
        beta = torch.full_like(alpha,fill_value=self._beta)


        return rho,a,b,c,alpha,beta

    def __call__(
        self,
        phi,
        rho_unconstrained,
        a_unconstrained,
        b_unconstrained,
        c_unconstrained,
        alpha_unconstrained,
        ret_logabsdet=True,
    ):

        rho,a,b,c,alpha,beta = self.constrain_params(
            rho_unconstrained,
            a_unconstrained,
            b_unconstrained,
            c_unconstrained,
            alpha_unconstrained
        )

        phi = bring_back_to_u1(phi,a=a,b=b,c=c,alpha=alpha,beta=beta)


        if ret_logabsdet:

            phi_out,ld = circular_bump(
                            phi=phi,
                            rho=rho,a=a,b=b,c=c,alpha=alpha,beta=beta,
                            ret_logabsdet=ret_logabsdet,
                        )
            phi_out = bring_back_to_u1(phi,a=a,b=b,c=c,alpha=alpha,beta=beta)

            return phi_out,ld
        
        else:

            phi_out = circular_bump(
                        phi=phi,
                        rho=rho,a=a,b=b,c=c,alpha=alpha,beta=beta,
                        ret_logabsdet=ret_logabsdet,
                        )
            phi_out = bring_back_to_u1(phi,a=a,b=b,c=c,alpha=alpha,beta=beta)

            return phi_out



    def inverse(
        self,
        phi,
        rho_unconstrained,
        a_unconstrained,
        b_unconstrained,
        c_unconstrained,
        alpha_unconstrained,
        ret_logabsdet=True,
    ):
        rho,a,b,c,alpha,beta = self.constrain_params(
            rho_unconstrained,
            a_unconstrained,
            b_unconstrained,
            c_unconstrained,
            alpha_unconstrained
        )


        phi = bring_back_to_u1(phi,a=a,b=b,c=c,alpha=alpha,beta=beta)
        phi_out = NumericalInverse.apply(phi, self.inverse_fn_params, rho,a,b,c,alpha,beta)
        phi_out = bring_back_to_u1(phi,a=a,b=b,c=c,alpha=alpha,beta=beta)

        if ret_logabsdet:
            _, ld = circular_bump(
                    phi=phi,
                    rho=rho,a=a,b=b,c=c,alpha=alpha,beta=beta,
                    ret_logabsdet=ret_logabsdet,
                )

            return phi_out, -ld
        else:
            return phi_out