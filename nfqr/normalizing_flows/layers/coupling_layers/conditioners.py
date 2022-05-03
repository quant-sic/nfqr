from xml.etree.ElementInclude import include

import torch
from torch.nn import Module

from nfqr.normalizing_flows.nets.nets import NET_REGISTRY
from nfqr.registry import StrRegistry

CONDITIONER_REGISTRY = StrRegistry("conditioners")


@CONDITIONER_REGISTRY.register("u1")
class ConditionerU1(Module):
    def __init__(self, dim_in, dim_out, expressivity, num_splits, net_config):
        super(ConditionerU1, self).__init__()

        self.net = NET_REGISTRY[net_config.net_type](
            in_size=dim_in * 2,
            out_size=dim_out,
            out_channels=expressivity * num_splits,
            **dict(net_config)
        )
        self.expressivity = expressivity

    def forward(self, z):

        z_u1 = torch.stack(
            [
                torch.cos(z),
                torch.sin(z),
            ],
            -1,
        ).view(*z.shape[:-1], -1)

        out = self.net(z_u1)
        h_pars = torch.split(out, self.expressivity, dim=-1)

        return h_pars
