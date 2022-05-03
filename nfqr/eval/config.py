from typing import Dict, Optional, Union

from nfqr.config import BaseConfig
from nfqr.target_systems import OBSERVABLE_REGISTRY


class EvalConfig(BaseConfig):

    _name: str = "eval_result"

    nip: Optional[Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]]
    nmcmc: Optional[
        Dict[Union[str, OBSERVABLE_REGISTRY.enum], Union[float, Dict[str, float]]]
    ]

    exact_sus: Optional[float]
