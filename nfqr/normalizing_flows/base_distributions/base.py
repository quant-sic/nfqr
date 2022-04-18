from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass
