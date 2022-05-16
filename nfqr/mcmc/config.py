from typing import Dict, List, Optional

from pydantic import validator

from nfqr.globals import REPO_ROOT
from nfqr.config import BaseConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY
from nfqr.target_systems import OBSERVABLE_REGISTRY

MLMC_PATH = REPO_ROOT / "mlmc"
MLMC_PATH.mkdir(exist_ok=True)


class MCMCConfig(BaseConfig):

    _name: str = "mcmc_result"

    observables: List[OBSERVABLE_REGISTRY.enum]
    hmc_type: HMC_REGISTRY.enum

    acceptance_rate: float
    n_steps: int
    obs_stats: Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]

    sus_exact:Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v
