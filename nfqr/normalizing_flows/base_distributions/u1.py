from math import pi
from typing import List, Tuple, Union
from venv import create

import torch
from pydantic import BaseModel
from torch.distributions import VonMises
from torch.nn import Module, parameter

from nfqr.normalizing_flows.base_distributions.base import BaseDistribution
from nfqr.normalizing_flows.misc.constraints import (  # nf_constraints_alternative,
    nf_constraints_standard,
)
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

U1_BASE_DIST_REGISTRY = StrRegistry("u1")
logger = create_logger(__name__)


@U1_BASE_DIST_REGISTRY.register("uniform")
class UniformBaseDistribution(BaseDistribution, Module):
    def __init__(
        self, dim=Tuple[int], left: float = 0.0, right: float = 2 * pi, **kwargs
    ) -> None:
        super().__init__()

        self.dim = dim

        # define parameters, such that .to method of module moves distribution to device
        self.left = parameter.Parameter(torch.tensor(left), requires_grad=False)
        self.right = parameter.Parameter(torch.tensor(right), requires_grad=False)

        self.dist = torch.distributions.uniform.Uniform(self.left, self.right)

    def sample(self, size):
        return self.dist.sample(sample_shape=(*size, *self.dim))

    def log_prob(self, value):
        try:
            return self.dist.log_prob(value)
        except:
            logger.info(f"min {value.min()}, max {value.max()}")
            raise


@U1_BASE_DIST_REGISTRY.register("von_mises")
class VonMisesBaseDistribution(BaseDistribution, Module):
    def __init__(
        self,
        dim: List[int],
        loc_requires_grad: bool = False,
        concentration_requires_grad: bool = False,
        loc: Union[None, List[float]] = None,
        concentration: Union[None, List[float]] = None,
        **kwargs,
    ) -> None:
        super(VonMisesBaseDistribution, self).__init__()

        self.dim = dim

        if loc is None or concentration is None:

            loc = torch.full(size=self.dim, fill_value=pi)
            concentration_unconstrained = torch.full(size=self.dim, fill_value=0.1)
            self.expand_sample_shape = False

        if not isinstance(loc, torch.Tensor):
            if isinstance(loc, float):
                loc = [loc]
            loc = torch.tensor(loc)

        self.constraint_transform = nf_constraints_standard(
            VonMises.arg_constraints["concentration"]
        )

        if not isinstance(concentration, torch.Tensor):
            if isinstance(concentration, float):
                concentration = [concentration]
            concentration_unconstrained = self.constraint_transform.inv(
                torch.tensor(concentration)
            )

        self.loc = parameter.Parameter(loc, requires_grad=loc_requires_grad)
        self.concentration_unconstrained = parameter.Parameter(
            concentration_unconstrained,
            requires_grad=concentration_requires_grad,
        )

    @classmethod
    def all_pars_joint(
        cls,
        dim: List[int],
        loc_requires_grad: bool = False,
        concentration_requires_grad: bool = False,
        **kwargs,
    ):

        loc = [pi]
        concentration = [0.1]

        return cls(
            dim=dim,
            loc_requires_grad=loc_requires_grad,
            concentration_requires_grad=concentration_requires_grad,
            loc=loc,
            concentration=concentration,
        )

    @property
    def logging_parameters(self):
        pars_dict = {}
        if len(self.loc)==1:
            pars_dict["loc"]=self.loc
        if len(self.concentration)==1:
            pars_dict["concentration"]=self.concentration

        return pars_dict

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
