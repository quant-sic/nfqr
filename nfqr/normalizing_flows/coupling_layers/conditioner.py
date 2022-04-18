import torch
from torch.nn import Module


class ConditionerU1(Module):
    def __init__(self, net, dim_in, dim_out, expressivity, num_splits, **net_kwargs):
        super(ConditionerU1, self).__init__()

        self.net = net(
            in_size=dim_in * 2,
            out_size=dim_out,
            out_channels=expressivity * num_splits,
            **net_kwargs
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
