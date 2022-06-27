from math import pi
from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from .observable import TopologicalCharge
from nfqr.registry import StrRegistry

ROTOR_TRAJECTORIES_REGISTRY = StrRegistry("qr")


@ROTOR_TRAJECTORIES_REGISTRY.register("classical")
class DiscreteClassicalRotorTrajectorySampler(object):
    def __init__(
        self, dim, batch_size=1, k_var=1.0, noise_std=0, k: int = None, **kwargs
    ):

        self.dim = dim
        self.k = k

        if k is None:
            self.probs = np.exp(
                -(np.arange(-dim[0] + 1, dim[0], 1) ** 2) / (2 * k_var)
            ) / np.sqrt(2 * pi * k_var)

            self.probs = self.probs / self.probs.sum()

        self.k_range = TopologicalCharge.k_range(self.dim)

        if self.k not in self.k_range:
            raise ValueError(
                f"k must be in {{{self.k_range[0]},...,{self.k_range[-1]}}}"
            )

        self.noise_std = noise_std
        self.batch_size = batch_size

    @property
    def data_specs(self):
        return {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "k": self.k,
            "target_system": "qr",
            "data_sampler": "rotor_classical",
        }

    def sample(self, device):

        if self.k is None:
            self.k = np.random.choice(
                self.k_range, size=self.batch_size, replace=True, p=self.probs
            )

        v_0 = torch.tensor([self.k], device=device, dtype=torch.float32) * (
            2 * pi / self.dim[0]
        )
        phi0 = torch.rand(self.batch_size, device=device) * 2 * pi
        noise = torch.rand(self.batch_size, self.dim[0], device=device) * self.noise_std

        phi = (
            -v_0[:, None] * torch.arange(0, self.dim[0], device=device)[None, :]
            + phi0[:, None]
            + noise
        ) % (2 * pi)

        return phi


@ROTOR_TRAJECTORIES_REGISTRY.register("hot")
class HotRotorTrajectoriesSampler(object):
    def __init__(self, dim, batch_size=1, **kwargs) -> None:
        self.dim = dim
        self.batch_size = batch_size

    @property
    def data_specs(self):
        return {
            "dim": self.dim,
            "target_system": "qr",
            "data_sampler": "rotor_hot",
        }

    def sample(self, device):
        config = torch.rand(self.batch_size, *self.dim, device=device) * 2 * pi

        return config


class RotorTrajectorySamplerConfig(BaseModel):

    traj_type: ROTOR_TRAJECTORIES_REGISTRY.enum
    dim: List[int]
    k: Optional[int]
    batch_size: Optional[int] = 1
