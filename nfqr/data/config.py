from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from nfqr.mcmc.config import MCMCConfig


class ConditionConfig(BaseModel):

    _name: str = "condition_config"

    params: Optional[Dict[str, Union[List[float], float, str]]] = None


class MCMCSamplerConfig(BaseModel):

    _name: str = "mcmc_sampler_config"

    mcmc_config: MCMCConfig
    num_batches: Optional[int] = 1
    condition_config: ConditionConfig
    batch_size: int
