from typing import List, Literal, Tuple, Union

from pydantic import BaseModel

from nfqr.normalizing_flows.base_distributions.config import BaseDistConfig
from nfqr.normalizing_flows.layers.config import LayersConfig


class FlowConfig(BaseModel):

    flow_type: str
    domain: Literal["u1"] = "u1"
    base_dist_config: BaseDistConfig
    layer_chain_config: Union[None, LayersConfig]
