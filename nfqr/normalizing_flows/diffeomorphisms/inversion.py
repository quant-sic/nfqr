import torch


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
