from math import pi

import torch
from torch.distributions import VonMises
from torch.nn import Module, parameter

from nfqr.normalizing_flows.base_distributions.base import BaseDistribution
from nfqr.normalizing_flows.misc.constraints import (
    nf_constraints_alternative,
    nf_constraints_standard,
)


class UniformBaseDistribution(BaseDistribution, Module):
    def __init__(
        self, dim: torch.Size, left: float = 0.0, right: float = 2 * pi
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dist = torch.distributions.uniform.Uniform(left, right)

    def sample(self, size):
        return self.dist.sample(sample_shape=(*size, *self.dim))

    def log_prob(self, value):
        return self.dist.log_prob(value)


class VonMisesBaseDistribution(BaseDistribution, Module):
    def __init__(
        self,
        dim: torch.Size,
        loc_requires_grad: bool = True,
        concentration_requires_grad: bool = True,
        loc: torch.Tensor = None,
        concentration_unconstrained: torch.Tensor = None,
    ) -> None:
        super(VonMisesBaseDistribution, self).__init__()

        self.dim = dim

        if loc is None or concentration_unconstrained is None:

            loc = torch.full(size=dim, fill_value=pi)
            concentration_unconstrained = torch.full(size=dim, fill_value=0.1)
            self.expand_sample_shape = False

        self.loc = parameter.Parameter(loc, requires_grad=loc_requires_grad)
        self.concentration_unconstrained = parameter.Parameter(
            concentration_unconstrained, requires_grad=concentration_requires_grad
        )
        self.constraint_transform = nf_constraints_standard(
            VonMises.arg_constraints["concentration"]
        )

    @classmethod
    def all_pars_joint(
        cls,
        dim: torch.Size,
        loc_requires_grad: bool = True,
        concentration_requires_grad: bool = True,
    ):

        loc = torch.tensor([pi])
        concentration_unconstrained = torch.tensor([0.1])

        return cls(
            dim,
            loc=loc,
            concentration_unconstrained=concentration_unconstrained,
            loc_requires_grad=loc_requires_grad,
            concentration_requires_grad=concentration_requires_grad,
        )

    @property
    def concentration(self):
        return self.constraint_transform(self.concentration_unconstrained)

    @property
    def dist(self):
        return VonMises(self.loc, self.concentration).expand(self.dim)

    @torch.no_grad()
    def sample(self, size: int) -> torch.Tensor:
        dist = self.dist
        return dist.sample(size) + pi

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        dist = self.dist
        return dist.log_prob(value - pi)


# class WrappedCauchy(BaseDistribution,Module):

#     pass
