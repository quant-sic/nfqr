import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, TypeVar, Union

from pydantic import root_validator, validator

from nfqr.config import BaseConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.target_systems import OBSERVABLE_REGISTRY, ActionConfig
from nfqr.utils import DimsNotMatchingError, set_par_list_or_dict

from .cluster import CLUSTER_REGISTRY

ConfigType = TypeVar("ConfigType", bound="MCMCConfig")


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
    n_replicas: Optional[int] = 1
    n_samples_at_a_time: Optional[int] = 10000

    initial_config_sampler_config: InitialConfigSamplerConfig

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

    @root_validator(pre=True)
    @classmethod
    def add_dims(cls, values):

        """
        Adds dims to sub configs.
        """

        dim = values["dim"]

        def set_dim(key, list_or_dict):

            if key in ("trajectory_sampler_config",):
                if "dim" not in list_or_dict[key]:

                    list_or_dict[key]["dim"] = dim
                else:
                    if not list_or_dict[key]["dim"] == dim:
                        raise DimsNotMatchingError(
                            dim,
                            list_or_dict[key]["dim"],
                            "Dim of top level ({}) and {} ({}) do not match".format(
                                dim, key, list_or_dict[key]["dim"]
                            ),
                        )

            return list_or_dict

        set_par_list_or_dict(values, set_fn=partial(set_dim))

        return values


class MCMCResult(BaseConfig):

    _name: str = "mcmc_result"

    mcmc_config: MCMCConfig
    acceptance_rate: float
    obs_stats: Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]

    sus_exact: Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v
