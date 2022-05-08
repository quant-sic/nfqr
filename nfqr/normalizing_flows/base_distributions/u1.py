from math import pi
from typing import List, Tuple, Union

import torch
from pydantic import BaseModel
from torch.distributions import VonMises
from torch.nn import Module, parameter

from nfqr.normalizing_flows.base_distributions.base import BaseDistribution
from nfqr.normalizing_flows.misc.constraints import (  # nf_constraints_alternative,
    nf_constraints_standard,
)
from nfqr.registry import StrRegistry

U1_BASE_DIST_REGISTRY = StrRegistry("u1")


@U1_BASE_DIST_REGISTRY.register("uniform")
class UniformBaseDistribution(BaseDistribution, Module):
    def __init__(
        self, dim=Tuple[int], left: float = 0.0, right: float = 2 * pi, **kwargs
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dist = torch.distributions.uniform.Uniform(left, right)

    def sample(self, size):
        return self.dist.sample(sample_shape=(*size, *self.dim))

    def log_prob(self, value):
        return self.dist.log_prob(value)


@U1_BASE_DIST_REGISTRY.register("von_mises")
class VonMisesBaseDistribution(BaseDistribution, Module):
    def __init__(
        self,
        dim: Tuple[int],
        loc_requires_grad: bool = False,
        concentration_requires_grad: bool = False,
        loc: Union[None, List[float]] = None,
        concentration_unconstrained: Union[None, List[float]] = None,
        **kwargs
    ) -> None:
        super(VonMisesBaseDistribution, self).__init__()

        self.dim = dim

        if loc is None or concentration_unconstrained is None:

            loc = torch.full(size=self.dim, fill_value=pi)
            concentration_unconstrained = torch.full(size=self.dim, fill_value=0.1)
            self.expand_sample_shape = False

        elif loc is None or concentration_unconstrained is None:
            loc = torch.tensor(loc)
            concentration_unconstrained = torch.tensor(concentration_unconstrained)

        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)

        if not isinstance(concentration_unconstrained, torch.Tensor):
            concentration_unconstrained = torch.tensor(concentration_unconstrained)

        self.loc = parameter.Parameter(loc, requires_grad=loc_requires_grad)
        self.concentration_unconstrained = parameter.Parameter(
            concentration_unconstrained,
            requires_grad=concentration_requires_grad,
        )
        self.constraint_transform = nf_constraints_standard(
            VonMises.arg_constraints["concentration"]
        )

    @classmethod
    def all_pars_joint(
        cls,
        dim: Tuple[int],
        loc_requires_grad: bool = False,
        concentration_requires_grad: bool = False,
        **kwargs
    ):

        loc = [pi]
        concentration_unconstrained = [0.1]

        return cls(
            dim=dim,
            loc_requires_grad=loc_requires_grad,
            concentration_requires_grad=concentration_requires_grad,
            loc=loc,
            concentration_unconstrained=concentration_unconstrained,
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


U1_BASE_DIST_REGISTRY.register(
    "von_mises_joint", VonMisesBaseDistribution.all_pars_joint
)


# class WrappedCauchy(BaseDistribution,Module):

#     pass

# https://github.com/jasonlaska/spherecluster
