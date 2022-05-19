from abc import ABC, abstractmethod

from pydantic import BaseModel
from pyparsing import Optional


class Action(ABC):
    @abstractmethod
    def evaluate(self, field):
        pass

    @abstractmethod
    def map_to_range(self, config):
        pass


class ActionConfig(BaseModel):

    beta: Optional[float]
