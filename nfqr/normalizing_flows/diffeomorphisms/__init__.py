from nfqr.registry import JointStrRegistry

from .u1 import U1_DIFFEOMORPHISM_REGISTRY
from .config import DiffeomorphismConfig

DIFFEOMORPHISMS_REGISTRY = JointStrRegistry(
    "diffeomorphisms", (U1_DIFFEOMORPHISM_REGISTRY,)
)
