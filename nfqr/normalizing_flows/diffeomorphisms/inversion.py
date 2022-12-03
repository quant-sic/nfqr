from functools import partial

import torch
from torch.autograd import Function, grad
from nfqr.utils import create_logger

logger = create_logger(__name__)

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def bisection_invert(f, out, left, right, tol=1e-4, nmax=1000):
    """
    Performs inversion using bisection search. Up to tolerance tol with a maximum number of tries nmax
    """
    assert left < right

    with torch.no_grad():

        left_ = torch.full(out.shape, left, device=out.device)
        right_ = torch.full(out.shape, right, device=out.device)

        for _ in range(nmax):

            c_ = (right_ + left_) / 2
            out_c = f(c_)

            diff = out - out_c
            if torch.max(torch.abs(diff)) < tol:
                return c_

            if torch.all((c_ == left_) + (c_ == right_)):
                return c_

            mask = (diff > 0).float()
            left_ = mask * c_ + (1 - mask) * left_
            right_ = (1 - mask) * c_ + mask * right_

        raise ValueError(
            f"bisection search has not converged within {nmax} steps up to a tolerance {tol}"
        )


class NumericalInverse(Function):
    @staticmethod
    def forward(ctx, x, fn, *args):
        func_partial = partial(
            fn["function"],
            **{name: args[idx] for idx, name in enumerate(fn["args"])},
            **fn["kwargs"],
        )

        z = bisection_invert(
            func_partial,
            x,
            left=fn["left"],
            right=fn["right"],
            tol=1e-4,
        )

        ctx.fn = fn
        ctx.save_for_backward(z, *args)

        return z

    @staticmethod
    def backward(ctx, grad_z):

        with torch.enable_grad():

            z, *pars = map(
                lambda t: t.clone().detach(), ctx.saved_tensors
            )

            grad_x = None
            grad_pars = [None] * len(pars)

            if any(ctx.needs_input_grad):

                # get parameter indices for tensors that require grad
                required_grad_idx = list(
                    filter(
                        lambda idx: ctx.needs_input_grad[-len(pars) :][idx],
                        range(len(pars)),
                    )
                )

                x = ctx.fn["function"](
                    # if any tensor requires grad then z also needs to require it
                    z.requires_grad_(True),
                    **{name: pars[idx].requires_grad_(idx in required_grad_idx) for idx, name in enumerate(ctx.fn["args"])},
                    **ctx.fn["kwargs"],
                )

                (grad_x_inverse,) = grad(x, z, torch.ones_like(x), retain_graph=True)

                grad_x = grad_z * grad_x_inverse.pow(-1)

                grads = grad(x, [pars[idx] for idx in required_grad_idx], -grad_x)

                for enum_idx, idx in enumerate(required_grad_idx):
                    grad_pars[idx] = grads[enum_idx]

        return grad_x, None, *grad_pars
