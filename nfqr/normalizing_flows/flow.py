import torch
from numpy import pi
from torch.nn import Module, ModuleList

from nfqr.normalizing_flows.coupling_layers.coupling import ResCoupling
from nfqr.normalizing_flows.coupling_layers.utils import generate_splits
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class U1CouplingChain(Module):
    def __init__(
        self,
        size,
        net,
        coupling,
        split_type,
        num_layers,
        expressivity,
        coupling_specifiers,
        **kwargs
    ):

        super(U1CouplingChain, self).__init__()

        self.layers = ModuleList()
        self.size = size

        splits = generate_splits(split_type, num_layers, size)
        for (conditioner_mask, transformed_mask), coupling_specifier in zip(
            splits, coupling_specifiers
        ):
            if coupling_specifier == "residual":
                c_ = coupling(
                    conditioner_mask, transformed_mask, net, expressivity, **kwargs
                )
                c = ResCoupling(c_)

            else:
                c = coupling(
                    conditioner_mask, transformed_mask, net, expressivity, **kwargs
                )
            self.layers.append(c)

    def encode(self, x):

        assert (x >= 0.0).all() and (x <= (2 * pi)).all()

        # x.shape[0] extracts batch dim
        abs_log_det = torch.zeros(x.shape[0], device=x.device)

        for coupling in self.layers[::-1]:
            x, ld = coupling.encode(x)
            abs_log_det += ld

        return x, abs_log_det

    def decode(self, z):

        if not ((z >= 0.0) & (z <= (2 * pi))).all():
            logger.info(z)

        log_det = torch.zeros(z.shape[0], device=z.device)

        for coupling in self.layers:
            z, ld = coupling.decode(z)
            log_det += ld

        return z, log_det

    # def forward(self, sample, reverse=False):

    #     if reverse:
    #         x, log_det = self.decode(z=sample)
    #         return x, log_det
    #     else:
    #         z, log_det = self.encode(x=sample)
    #         return z, log_det

    def load(self, checkpoint, device):
        self.load_state_dict(torch.load(checkpoint, map_location=device)["net"])


class U1Flow(Module):
    def __init__(self, base_distribution, transform) -> None:
        super(U1Flow, self).__init__()

        self.base_distribution = base_distribution
        self.transform = transform

    def log_prob(self, x):

        z, abs_log_det = self.transform.encode(x)
        q_z = self.base_distribution.log_prob(z).sum(dim=-1)

        return q_z + abs_log_det

    def sample(self, size):

        x, _ = self.sample_with_abs_log_det(size)

        return x

    def sample_with_abs_log_det(self, size):

        z = self.base_distribution.sample(size)
        x, abs_log_det = self.transform.decode(z)

        q_x = self.base_distribution.log_prob(z).sum(dim=-1) - abs_log_det

        return x, q_x
