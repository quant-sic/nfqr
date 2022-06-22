import json
from typing import Dict, Union

import torch

from nfqr.target_systems import OBSERVABLE_REGISTRY
from nfqr.utils import create_logger

logger = create_logger(__name__)


class SampleCondition(object):
    def __init__(self, params: Union[Dict, None] = None) -> None:
        self.params = params

        self.save_keys = ("type", "target_system", "observable", "value")

        if self.params is None:
            self.evaluate = lambda sample: True
            self.repr = json.dumps(None)

        elif isinstance(self.params, dict) and self.params["type"] == "observable":
            self.evaluate = self.evaluate_observable
            self.observable_fn = OBSERVABLE_REGISTRY[params["target_system"]][
                params["observable"]
            ]().evaluate
            self.repr = json.dumps({k: self.params[k] for k in self.save_keys})

    @classmethod
    def from_str(cls, _str: str):
        params = json.loads(_str)
        return cls(params)

    def __eq__(self, other):
        if any(d is None for d in (self.params, other.params)):
            return all(d is None for d in (self.params, other.params)) or all(
                d is not None for d in (self.params, other.params)
            )
        else:
            return all(self.params[k] == other.params[k] for k in self.save_keys)

    def evaluate_observable(self, sample):
        return torch.round(self.observable_fn(sample)).item() in self.params["value"]

    def __repr__(self) -> str:
        return self.repr
