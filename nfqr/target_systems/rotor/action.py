import math
from typing import List, Optional

import jax.numpy as jnp
import torch
from pydantic import BaseModel

from nfqr.registry import StrRegistry
from nfqr.target_systems.action import ClusterAction

ROTOR_ACTION_REGISTRY = StrRegistry("qr")


@ROTOR_ACTION_REGISTRY.register("qr")
class QuantumRotor(ClusterAction):
    def __init__(
        self,
        beta: float,
        dim: List[int],
        mom_inertia: float = None,
        T: float = None,
        diffs=False,
    ) -> None:
        super().__init__()
        self._beta = beta
        self._T = T
        self._mom_inertia = mom_inertia
        self._dim = dim

        if beta is None and (mom_inertia is None or T is None or dim is None):
            raise ValueError("Either beta of mom_inertia and T and dim must be given")

        self.diffs = diffs

    @property
    def beta(self):
        if self._beta is not None:
            return self._beta
        else:
            self._beta = self._mom_inertia / (self._T / self._dim[0])
            return self._beta

    @beta.setter
    def beta(self, v):
        self._beta = v

    @classmethod
    def use_diffs(cls, beta):
        return cls(beta, diffs=True)

    @staticmethod
    def _get_diffs(config: torch.Tensor) -> torch.Tensor:
        return config - torch.roll(config, shifts=1, dims=-1)

    def evaluate(self, config: torch.Tensor) -> torch.Tensor:
        if not self.diffs:
            _config = self._get_diffs(config=config)
        else:
            _config = config

        return self.beta * (1.0 - torch.cos(_config)).sum(dim=-1)

    def evaluate_jnp(self, config):
        if not self.diffs:
            _config = jnp.roll(config, shift=1, axis=-1) - config
        else:
            _config = config

        return self.beta * (1.0 - jnp.cos(_config)).sum(axis=-1)

    @staticmethod
    def map_to_range(config: torch.Tensor) -> torch.Tensor:
        return config % (2 * math.pi)

    def bonding_prob(
        self, config_left: torch.Tensor, config_right: torch.Tensor, reflection
    ) -> torch.Tensor:
        S_ell = (
            2.0
            * self.beta
            * torch.cos(config_left - reflection)
            * torch.cos(config_right - reflection)
        )
        return 1.0 - S_ell.clamp(max=0.0).exp()

    def flip(self, config: torch.Tensor, reflection: torch.Tensor) -> torch.Tensor:

        angle = math.pi + 2 * reflection - config

        return self.map_to_range(angle)

    def force(self, config: torch.Tensor) -> torch.Tensor:

        force = self.beta * (
            torch.sin(config - torch.roll(config, shifts=-1, dims=-1))
            - torch.sin(torch.roll(config, shifts=1, dims=-1) - config)
        )

        return force


ROTOR_ACTION_REGISTRY.register("qr_diffs", QuantumRotor.use_diffs)


class QuantumRotorConfig(BaseModel):

    beta: Optional[float]
    dim: Optional[List[int]]
    T: Optional[float]
    mom_inertia: Optional[float]
