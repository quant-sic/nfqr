from abc import ABC, abstractmethod


class Observable(ABC):
    def __init__(self) -> None:
        pass

    def prepare(self, history):
        return history

    def postprocess(self, value):
        return value

    @abstractmethod
    def evaluate(self, config):
        raise NotImplementedError()
