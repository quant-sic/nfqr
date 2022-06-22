from pydantic import BaseModel

from nfqr.target_systems import TRAJECTORY_SAMPLER_REGISTRY
from nfqr.target_systems.rotor import RotorTrajectorySamplerConfig


class InitialConfigSampler(object):
    def __init__(
        self,
        trajectory_sampler_config: RotorTrajectorySamplerConfig,
        target_system,
    ) -> None:

        self.trajectory_sampler = TRAJECTORY_SAMPLER_REGISTRY[target_system][
            trajectory_sampler_config.traj_type
        ](**dict(trajectory_sampler_config))

    def sample(self, device):

        config = self.trajectory_sampler.sample(device)

        return config


class InitialConfigSamplerConfig(BaseModel):

    trajectory_sampler_config: RotorTrajectorySamplerConfig
    target_system: str = "qr"
