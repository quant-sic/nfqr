from typing import Literal, Union

from pydantic import BaseModel
from torch.nn import Module

from nfqr.normalizing_flows.base_distributions import BASE_DIST_REGISTRY, BaseDistConfig
from nfqr.normalizing_flows.layers import LayerChain, LayerChainConfig
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class FlowConfig(BaseModel):

    flow_type: str
    domain: Literal["u1"] = "u1"
    base_dist_config: BaseDistConfig
    layer_chain_config: Union[None, LayerChainConfig]


class BareFlow(Module):
    def __init__(
        self,
        base_dist_config: BaseDistConfig,
        layer_chain_config: LayerChainConfig,
        domain: Literal["u1"] = "u1",
        **kwargs
    ) -> None:
        super(BareFlow, self).__init__()
        self.base_distribution = BASE_DIST_REGISTRY[domain][base_dist_config.dist_type](
            **dict(base_dist_config)
        )
        if layer_chain_config is not None:
            self.transform = LayerChain(**dict(layer_chain_config))

    def log_prob(self, x):

        z, abs_log_det = self.transform.encode(x)
        q_z = self.base_distribution.log_prob(z).sum(dim=-1)

        return q_z + abs_log_det

    def sample(self, size):

        x, _ = self.sample_with_abs_log_det(size)

        return x

    def sample_with_abs_log_det(self, size):

        device = self.parameters().__next__().device
        z = self.base_distribution.sample(size).to(device)
        x, abs_log_det = self.transform.decode(z)

        q_x = self.base_distribution.log_prob(z).sum(dim=-1) - abs_log_det

        return x, q_x
