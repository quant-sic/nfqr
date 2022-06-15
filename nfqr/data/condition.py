import json
from typing import Dict, Union

import torch

from nfqr.target_systems import OBSERVABLE_REGISTRY
from nfqr.utils import create_logger

logger = create_logger(__name__)


class SampleCondition(object):
    def __init__(self, params: Union[Dict, None] = None) -> None:
        self.params = params

        if self.params is None:
            self.evaluate = lambda sample: True
            self.repr = "None"

        elif isinstance(self.params, dict) and self.params["type"] == "observable":
            self.evaluate = self.evaluate_observable
            self.observable_fn = OBSERVABLE_REGISTRY[params["target_system"]][
                params["observable"]
            ]().evaluate
            self.repr = json.dumps(
                {
                    k: self.params[k]
                    for k in ("type", "target_system", "observable", "value")
                }
            )

    def evaluate_observable(self, sample):
        return torch.round(self.observable_fn(sample)).item() in self.params["value"]

    def __repr__(self) -> str:
        return self.repr
