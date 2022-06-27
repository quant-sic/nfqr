from abc import ABC, abstractmethod


import torch


class Action(ABC):
    @abstractmethod
    def evaluate(self, field):
        pass

    @abstractmethod
    def map_to_range(self, config):
        pass

class ClusterAction(Action):
    @abstractmethod
    def bonding_prob(
        self, config_left: torch.Tensor, config_right: torch.Tensor, reflection
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def flip(self, config: torch.Tensor, reflection: torch.Tensor) -> torch.Tensor:
        pass


