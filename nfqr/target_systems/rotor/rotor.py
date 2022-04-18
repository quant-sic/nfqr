import math

import jax.numpy as jnp
import torch

from nfqr.target_systems.action import Action
from nfqr.target_systems.observable import Observable


class QuantumRotorDiffs(Action):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def evaluate(self, config: torch.Tensor) -> torch.Tensor:

        return self.beta * (1.0 - torch.cos(config)).sum(dim=-1)

    @staticmethod
    def map_to_range(config: torch.Tensor) -> torch.Tensor:
        return config % (2 * math.pi) - math.pi


class QuantumRotor(Action):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def evaluate(self, config: torch.Tensor) -> torch.Tensor:

        shifted_config = torch.roll(config, shifts=1, dims=-1)

        return self.beta * (1.0 - torch.cos(shifted_config - config)).sum(dim=-1)

    def evaluate_jnp(self, config):

        shifted_config = jnp.roll(config, shift=1, axis=-1)

        return self.beta * (1.0 - jnp.cos(shifted_config - config)).sum(axis=-1)

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

    @staticmethod
    def map_to_range(config: torch.Tensor) -> torch.Tensor:
        return config % (2 * math.pi)

    def force(self, config: torch.Tensor) -> torch.Tensor:

        force = self.beta * (
            torch.sin(config - torch.roll(config, -1, -1))
            - torch.sin(torch.roll(config, shifts=1, dim=-1) - config)
        )

        return force


class TopologicalCharge(Observable):
    def __init__(self):
        super().__init__()

    def evaluate(self, config):

        shifted_config = torch.roll(config, shifts=1, dims=-1)
        return (((shifted_config - config) + math.pi) % (2 * math.pi) - math.pi).sum(
            dim=-1
        ) / (2 * math.pi)


class TopologicalSusceptibility(Observable):
    def __init__(self):
        super().__init__()
        self.charge = TopologicalCharge()

    def evaluate(self, config):

        charge = self.charge.evaluate(config)
        susceptibility = charge**2
        return susceptibility
