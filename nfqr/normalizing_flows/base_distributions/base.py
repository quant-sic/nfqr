from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from pydantic import BaseModel


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass
