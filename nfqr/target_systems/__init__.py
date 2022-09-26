from itertools import chain

from nfqr.registry import JointStrRegistry

from .config import ActionConfig
from .rotor import (
    ROTOR_ACTION_REGISTRY,
    ROTOR_OBSERVABLE_REGISTRY,
    ROTOR_TRAJECTORIES_REGISTRY,
)

ACTION_REGISTRY = JointStrRegistry("action", (ROTOR_ACTION_REGISTRY,))
OBSERVABLE_REGISTRY = JointStrRegistry("obs", (ROTOR_OBSERVABLE_REGISTRY,))
TRAJECTORY_SAMPLER_REGISTRY = JointStrRegistry(
    "traj_sampler", (ROTOR_TRAJECTORIES_REGISTRY,)
)
