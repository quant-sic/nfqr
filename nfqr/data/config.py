from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from nfqr.mcmc.config import MCMCConfig
from nfqr.target_systems.rotor import RotorTrajectorySamplerConfig


class ConditionConfig(BaseModel):

    _name: str = "condition_config"

    params: Optional[Dict[str, Union[List[float], float, str]]] = None


class TrajectorySamplerConfig(BaseModel):

    _name: str = "trajectory_sampler_config"

    sampler_config: Union[MCMCConfig, RotorTrajectorySamplerConfig]
    num_batches: Optional[int] = 1
    condition_config: ConditionConfig
    sampler_batch_size: int


class PSamplerConfig(BaseModel):

    _name: str = "p_sampler_config"

    sampler_configs: List[TrajectorySamplerConfig]
    batch_size: int
    elements_per_dataset: int
    subset_distribution: List[float] = None
    num_workers: int
    num_batches: int = None
    shuffle: bool = True
    infinite: bool = True
