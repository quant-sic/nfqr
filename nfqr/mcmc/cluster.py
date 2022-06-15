import math
import random

import torch

from nfqr.mcmc.base import MCMC
from nfqr.registry import StrRegistry
from nfqr.target_systems import ACTION_REGISTRY
from nfqr.target_systems.action import ClusterAction
from nfqr.utils import create_logger

logger = create_logger(__name__)

CLUSTER_REGISTRY = StrRegistry("cluster")


@CLUSTER_REGISTRY.register("wolff")
class WolffCluster(MCMC):
    def __init__(
        self,
        action_config,
        action,
        dim,
        n_steps,
        observables,
        out_dir,
        n_traj_steps=1,
        n_burnin_steps=0,
        batch_size=1,
        target_system="qr",
        **kwargs,
    ):

        super(WolffCluster, self).__init__(
            n_steps=n_steps,
            observables=observables,
            target_system=target_system,
            out_dir=out_dir,
        )

        self.action = ACTION_REGISTRY[target_system][action](**dict(action_config))

        if not isinstance(self.action, ClusterAction):
            raise ValueError(
                f"Cluster Algorithm only accepts Cluster action. Action give is of type {type(self.action)}"
            )

        self.n_burnin_steps = n_burnin_steps
        self.dim = dim
        self.batch_size = batch_size
        self.n_traj_steps = n_traj_steps
        self.target_system = target_system

    @property
    def data_specs(self):
        return {
            "dim": self.dim,
            "beta": self.action.beta,
            "n_burnin_steps": self.n_burnin_steps,
            "n_traj_steps": self.n_traj_steps,
            "target_system": self.target_system,
            "data_sampler": "wolff_cluster",
        }

    @property
    def acceptance_rate(self):
        return self.n_accepted / self.n_current_steps

    def step(self, config=None, record_observables=True):
        for _ in range(self.n_traj_steps):
            self.cluster_update_batch(config)

        self.n_accepted += 1

        if record_observables:
            self.observables_rec.record_config(self.current_config[0])

    def cluster_update(self):

        config = self.current_config

        nlat = config.shape[1]
        start = random.randint(0, nlat - 1)
        reflect = (torch.rand(1) - 0.5) * 2 * math.pi

        marked = set([start])
        config[:, start] = self.action.flip(config[:, start], reflect)
        for direction in (-1, 1):
            index = start
            for _ in range(nlat):
                neighbor = (index + direction) % nlat
                if neighbor in marked:
                    break
                if (
                    self.action.bonding_prob(
                        config[:, index], config[:, neighbor], reflect
                    )
                    < torch.rand(1)
                ).all():
                    break
                marked.add(neighbor)
                config[:, neighbor] = self.action.flip(config[:, neighbor], reflect)
                index = neighbor

    def cluster_update_batch(self, config=None):

        if config is None:
            config = self.current_config

        start = torch.randint(size=(self.batch_size,), low=0, high=self.dim[0] - 1)

        reflect = torch.rand(size=(self.batch_size,)) * 2 * math.pi

        marked = torch.full_like(config, fill_value=False, dtype=bool)
        marked[range(self.batch_size), start] = True

        config[range(self.batch_size), start] = self.action.flip(
            config[range(self.batch_size), start], reflect
        )

        for direction in (-1, 1):
            index = start
            done_mask = torch.full((self.batch_size,), fill_value=False)

            for _ in range(self.dim[0]):
                neighbor = (index + direction) % self.dim[0]

                done_mask = (
                    done_mask
                    | marked[range(self.batch_size), neighbor].view(-1)
                    | (
                        self.action.bonding_prob(
                            config[range(self.batch_size), index],
                            config[range(self.batch_size), neighbor],
                            reflect,
                        )
                        < torch.rand(size=(self.batch_size,))
                    ).view(-1)
                )

                if done_mask.all():
                    break

                marked[range(self.batch_size), neighbor] = True

                config[~done_mask, neighbor[~done_mask]] = self.action.flip(
                    config[~done_mask, neighbor[~done_mask]], reflect[~done_mask]
                )

                index = neighbor
                

    def initialize(self):

        self.current_config = (
            torch.rand(self.batch_size, *self.dim, dtype=torch.float64) * 2 * math.pi
        )

        for _ in range(self.n_burnin_steps):
            self.cluster_update_batch()
