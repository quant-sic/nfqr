from typing import List, Tuple, Union

from pydantic import BaseModel

from nfqr.registry import JointStrRegistry

from .u1 import U1_BASE_DIST_REGISTRY

BASE_DIST_REGISTRY = JointStrRegistry("base_dist", (U1_BASE_DIST_REGISTRY,))


class BaseDistConfig(BaseModel):

    dist_type: BASE_DIST_REGISTRY.enum
    dim: Tuple[int]

    loc_requires_grad: bool = False
    concentration_requires_grad: bool = False
    loc: Union[None, List[float]] = None
    concentration_unconstrained: Union[None, List[float]] = None
