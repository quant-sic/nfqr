from typing import Union
from pydantic import BaseModel
from .u1 import U1_DIFFEOMORPHISM_REGISTRY,NCPConfig

class DiffeomorphismConfig(BaseModel):

    diffeomorphism_type: U1_DIFFEOMORPHISM_REGISTRY.enum
    specific_diffeomorphism_config: Union[NCPConfig,None]
