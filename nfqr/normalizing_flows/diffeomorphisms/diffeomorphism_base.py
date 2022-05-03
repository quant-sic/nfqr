from abc import ABC, abstractmethod


class Diffeomorphism(ABC):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

    @abstractmethod
    def constrain_params(self):
        pass

    @property
    @abstractmethod
    def num_pars(self):
        pass
