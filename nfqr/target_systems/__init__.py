from nfqr.registry import JointStrRegistry, StrRegistry

from .rotor.rotor import ROTOR_ACTION_REGISTRY, ROTOR_OBSERVABLE_REGISTRY

ACTION_REGISTRY = JointStrRegistry("action", (ROTOR_ACTION_REGISTRY,))
OBSERVABLE_REGISTRY = JointStrRegistry("obs", (ROTOR_OBSERVABLE_REGISTRY,))
