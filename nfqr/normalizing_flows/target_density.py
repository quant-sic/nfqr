import torch

from nfqr.target_systems.action import Action


class BoltzmannFactor(object):
    def __init__(self, action: Action) -> None:
        self.action = action

    def log_prob(self, value: torch.Tensor):
        return -self.action.evaluate(value)


class TargetDensity(object):
    def __init__(self, distribution) -> None:

        if not hasattr(distribution, "log_prob"):
            raise ValueError("Given distribution must implement a log_prob attribute")

        self.dist = distribution

    @classmethod
    def boltzmann_from_action(cls, action: Action):
        boltzmann_factor = BoltzmannFactor(action=action)
        return cls(boltzmann_factor)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)
