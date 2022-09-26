from typing import Union
from pydantic import BaseModel
from .u1 import NCPConfig

class DiffeomorphismConfig(BaseModel):

    diffeomorphism_type: str
    specific_diffeomorphism_config: Union[NCPConfig,None]
