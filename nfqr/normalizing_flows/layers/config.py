from typing import List, Tuple, Union

from pydantic import BaseModel

from nfqr.normalizing_flows.layers.coupling_layers.config import CouplingConfig
from nfqr.normalizing_flows.layers.coupling_layers.utils import SPLIT_TYPES


class LayersConfig(BaseModel):

    dim: Tuple[int]
    layers_config: Union[None, CouplingConfig, List[CouplingConfig]]
    split_type: SPLIT_TYPES
    num_layers: int
