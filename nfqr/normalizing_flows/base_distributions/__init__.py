from nfqr.registry import JointStrRegistry
from .base import BaseDistConfig
from .u1 import U1_BASE_DIST_REGISTRY

BASE_DIST_REGISTRY = JointStrRegistry("base_dist", (U1_BASE_DIST_REGISTRY,))
