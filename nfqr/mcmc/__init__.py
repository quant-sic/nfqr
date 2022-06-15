from nfqr.registry import JointStrRegistry

from .base import MCMC, get_mcmc_statistics
from .cluster import CLUSTER_REGISTRY
from .hmc.hmc import HMC_REGISTRY

MCMC_REGISTRY = JointStrRegistry("mcmc", (CLUSTER_REGISTRY, HMC_REGISTRY))
