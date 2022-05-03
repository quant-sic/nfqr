from typing import Literal

from pydantic import BaseModel, Field

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.nets.config import NetConfig

from .couplings import COUPLING_TYPES


class CouplingConfig(BaseModel):

    domain: Literal["u1"] = "u1"
    coupling_type: COUPLING_TYPES.enum = Field(...)
    diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum
    expressivity: int
    net_config: NetConfig

    # validators ..
