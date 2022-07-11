from typing import Literal

import torch
from torch.nn import Module

from nfqr.normalizing_flows.diffeomorphisms import DIFFEOMORPHISMS_REGISTRY
from nfqr.normalizing_flows.nets import NetConfig
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

from nfqr.normalizing_flows.layers.conditioners import CONDITIONER_REGISTRY

logger = create_logger(__name__)

AR_LAYER_TYPES = StrRegistry("ar_layer_types")

class AutoregressiveLayer(Module):
    def __init__(
        self,
        dim,
        diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum,
        expressivity: int,
        net_config: NetConfig,
        domain: Literal["u1"] = "u1",
        **kwargs,
    ) -> None:
        super(AutoregressiveLayer, self).__init__()

        self.diffeomorphism = DIFFEOMORPHISMS_REGISTRY[domain][diffeomorphism]()
        self.expressivity=expressivity
        self.net_config = net_config
        self.domain = domain
        self.dim = dim

    def decode(self, z):
        pass

    def encode(self, x):
        pass


@AR_LAYER_TYPES.register("iterative")
class IterativeARLayer(AutoregressiveLayer, Module):
    def __init__(self, dim, diffeomorphism: DIFFEOMORPHISMS_REGISTRY.enum, expressivity: int, net_config: NetConfig, domain: Literal["u1"] = "u1", **kwargs) -> None:
        super().__init__(dim,diffeomorphism, expressivity, net_config, domain, **kwargs)

        if not len(dim)==1:
            raise ValueError("Layer not yet constructed for multidimensional input dim")

        self.conditioners = []
        for idx in range(1,dim[0]):
            self.conditioners += [CONDITIONER_REGISTRY[domain](
                dim_in=idx,
                dim_out=1,
                expressivity=expressivity,
                num_splits=self.diffeomorphism.num_pars,
                net_config=net_config,
            )]

    def decode(self, z):
        
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])

        for idx in range(1,z.shape[-1]):
            
            unconstrained_params = self.conditioners[idx](z[...,:idx])
            x[...,idx],ld = self.diffeomorphism(
                z[...,idx], *unconstrained_params, ret_logabsdet=True
            )

            log_det += ld

        return x,log_det

    def encode(self, x):
        
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.shape[0])

        for idx in range(1,x.shape[-1]):
            
            unconstrained_params = self.conditioners[idx](x[...,:idx])
            z[...,idx],ld = self.diffeomorphism.inverse(
                z[...,idx], *unconstrained_params, ret_logabsdet=True
            )

            log_det += ld

        return z,log_det



