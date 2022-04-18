from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def evaluate(self, field):
        pass

    @abstractmethod
    def map_to_range(self, config):
        pass
