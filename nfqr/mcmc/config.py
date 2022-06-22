import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, TypeVar, Union

from pydantic import validator

from nfqr.config import BaseConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig

from .cluster import CLUSTER_REGISTRY

ConfigType = TypeVar("ConfigType", bound="MCMCConfig")


class MCMCResult(BaseConfig):

    _name: str = "mcmc_result"

    observables: List[OBSERVABLE_REGISTRY.enum]
    hmc_type: HMC_REGISTRY.enum

    acceptance_rate: float
    n_steps: int
    obs_stats: Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]

    sus_exact: Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v


class MCMCConfig(BaseConfig):

    _name: str = "mcmc_config"

    mcmc_type: Union[CLUSTER_REGISTRY.enum, HMC_REGISTRY.enum]
    mcmc_alg: Literal["cluster", "hmc"]

    observables: List[OBSERVABLE_REGISTRY.enum]
    n_steps: int
    dim: List[int]
    action_config: ActionConfig
    n_burnin_steps: int
    out_dir: Union[str, Path]
    n_traj_steps: Optional[int] = 20
    step_size: Optional[float] = 0.01
    autotune_step: Optional[bool] = True
    hmc_engine: Optional[Literal["cpp_batch", "cpp_single", "python"]] = "cpp_single"
    batch_size: Optional[int] = 1
    n_samples_at_a_time: Optional[int] = 10000
    target_system: Optional[ACTION_REGISTRY.enum] = "qr"
    action: Optional[ACTION_REGISTRY.enum] = "qr"

    initial_config_sampler_config: Optional[InitialConfigSamplerConfig]

    task_parameters: Union[List[str], None] = None

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v

    @classmethod
    def from_directory_for_task(
        cls: Type[ConfigType], directory: Union[str, Path], task_id
    ) -> ConfigType:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        def set_task_par(_dict):
            for key, value in _dict.items():
                if isinstance(value, dict):
                    _dict[key] = set_task_par(value)

                if key in raw_config["task_parameters"]:
                    _dict[key] = _dict[key][task_id]

            return _dict

        if raw_config["task_parameters"] is not None:
            raw_config = set_task_par(raw_config)

        raw_config["out_dir"] = directory / f"mcmc/task_{task_id}"

        return cls(**raw_config)
