from typing import List, Tuple, Union

from pydantic import BaseModel

from nfqr.normalizing_flows.base_distributions import BASE_DIST_REGISTRY


class BaseDistConfig(BaseModel):

    dist_type: BASE_DIST_REGISTRY.enum
    dim: Tuple[int]

    loc_requires_grad: bool = False
    concentration_requires_grad: bool = False
    loc: Union[None, List[float]] = None
    concentration_unconstrained: Union[None, List[float]] = None
