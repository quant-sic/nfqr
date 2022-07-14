from typing import Literal, Union

from pydantic import BaseModel

from nfqr.normalizing_flows.layers.layer_splits import LayerSplitConfig
from nfqr.registry import JointStrRegistry

from .autoregressive_layers import AR_LAYER_REGISTRY, ARLayerConfig
from .coupling_layers import COUPLING_LAYER_REGISTRY, CouplingConfig

LAYER_REGISTRY = JointStrRegistry("layer", (COUPLING_LAYER_REGISTRY, AR_LAYER_REGISTRY))


class LayerConfig(BaseModel):

    layer_type: Literal["ar_layer", "coupling_layer"]
    specific_layer_config: Union[CouplingConfig, ARLayerConfig]
    layer_split_config: Union[LayerSplitConfig, None]
    num_layers: int
