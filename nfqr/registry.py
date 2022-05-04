from itertools import chain

from strenum import StrEnum


class StrRegistry(object):
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

        super(StrRegistry, self).__init__()

    @property
    def name(self):
        return self._name

    def __dict__(self):
        return self._registry

    def register(self, name, factory=None):

        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(name=name, factory=factory)

        self._registry[name] = factory

        return factory

    def __getitem__(self, name):

        try:
            factory = self._registry[name]
        except KeyError:
            raise NotImplementedError(
                f"No Factory registered for name {name}"
            ) from None
        return factory

    @property
    def enum(self):
        return StrEnum(self._name, list(self._registry.keys()))


class JointStrRegistry(object):
    def __init__(self, name, registries) -> None:

        self._name = name
        self._registry = {reg.name: reg for reg in registries}

    def __getitem__(self, name):
        return self._registry[name]

    @property
    def enum(self):
        return StrEnum(
            self._name,
            list(
                chain.from_iterable(
                    (v.__dict__().keys() for v in self._registry.values())
                )
            ),
        )
