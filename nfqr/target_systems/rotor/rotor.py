import math

import jax.numpy as jnp
import torch

from nfqr.registry import StrRegistry
from nfqr.target_systems.action import Action
from nfqr.target_systems.observable import Observable

ROTOR_OBSERVABLE_REGISTRY = StrRegistry("qr")
ROTOR_ACTION_REGISTRY = StrRegistry("qr")


@ROTOR_ACTION_REGISTRY.register("qr")
class QuantumRotor(Action):
    def __init__(self, beta: float, diffs=False) -> None:
        super().__init__()
        self.beta = beta
        self.diffs = diffs

    @classmethod
    def use_diffs(cls, beta):
        return cls(beta, diffs=True)

    def evaluate(self, config: torch.Tensor) -> torch.Tensor:
        if not self.diffs:
            _config = torch.roll(config, shifts=1, dims=-1) - config
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
        return config % (2 * math.pi) - math.pi

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
            torch.sin(config - torch.roll(config, -1, -1))
            - torch.sin(torch.roll(config, shifts=1, dim=-1) - config)
        )

        return force


ROTOR_ACTION_REGISTRY.register("qr_diffs", QuantumRotor.use_diffs)


@ROTOR_OBSERVABLE_REGISTRY.register("Q")
class TopologicalCharge(Observable):
    def __init__(self, diffs=False):
        super().__init__()
        self.diffs = diffs

    @classmethod
    def use_diffs(cls):
        return cls(diffs=True)

    def evaluate(self, config):
        if not self.diffs:
            _config = torch.roll(config, shifts=1, dims=-1) - config
        else:
            _config = config

        return ((_config + math.pi) % (2 * math.pi) - math.pi).sum(dim=-1) / (
            2 * math.pi
        )


ROTOR_OBSERVABLE_REGISTRY.register("Q_diffs", TopologicalCharge.use_diffs)


@ROTOR_OBSERVABLE_REGISTRY.register("Chi_t")
class TopologicalSusceptibility(Observable):
    def __init__(self, diffs=False):
        super().__init__()

        self.charge = TopologicalCharge(diffs=diffs)

    @classmethod
    def use_diffs(cls):
        return cls(diffs=True)

    def evaluate(self, config):

        charge = self.charge.evaluate(config)
        susceptibility = charge**2
        return susceptibility


ROTOR_OBSERVABLE_REGISTRY.register("Chi_t_diffs", TopologicalSusceptibility.use_diffs)
