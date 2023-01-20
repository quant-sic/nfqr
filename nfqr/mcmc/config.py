import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

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

    n_repeat: int = 1
    min_error: Optional[float] = None

    mcmc_type: str
    mcmc_alg: Literal["cluster", "hmc"]

    observables: List[str]
    n_steps: Union[int, List[int]]
    dim: List[int]
    action_config: ActionConfig
    n_burnin_steps: int
    out_dir: Path
    n_traj_steps: Optional[int] = 20
    step_size: Optional[float] = 0.01
    autotune_step: Optional[bool] = True
    hmc_engine: Optional[Literal["cpp_batch", "cpp_single", "python"]] = "cpp_single"
    n_replicas: Optional[int] = 1
    n_samples_at_a_time: Optional[int] = 10000
    int_time: Optional[Union[float, None]] = None

    initial_config_sampler_config: InitialConfigSamplerConfig

    task_parameters: Union[List[str], None] = None

    stats_step_interval: int = 100
    max_stats_eval: int = 1e6
    stats_skip_steps: int = 1

    stats_method: Literal["wolff", "blocked"] = "wolff"

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v

    @classmethod
    def get_num_tasks(cls: Type[ConfigType], directory: Union[str, Path]) -> int:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        num_pars_dict = {}

        def fill_num_pars_dict(key, list_or_dict):

            if key in raw_config["task_parameters"]:
                try:
                    num_pars_dict[key] = len(list_or_dict[key])
                except TypeError:
                    raise RuntimeError(
                        "Len could not be evaluated for {}".format(list_or_dict[key])
                    )
            return list_or_dict

        if raw_config["task_parameters"] is not None:
            raw_config = set_par_list_or_dict(
                raw_config, set_fn=partial(fill_num_pars_dict)
            )

        # check for inconsistencies in task array setup and config
        if not len(set(num_pars_dict.values())) <= 1:
            raise ValueError(
                f"Inconsistent number of tasks for parameters. {num_pars_dict}"
            )
        else:
            num_pars = (
                list(num_pars_dict.values())[0] if list(num_pars_dict.values()) else 1
            )

        return num_pars

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

            if key in ("trajectory_sampler_config", "specific_action_config"):
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

    @root_validator(pre=True)
    @classmethod
    def add_n_replicas(cls, values):

        """
        Adds dims to sub configs.
        """

        n_replicas = values.get("n_replicas", 1)

        def set_n_replicas(key, list_or_dict):

            if key in ("trajectory_sampler_config",):
                if "n_replicas" not in list_or_dict[key]:

                    list_or_dict[key]["n_replicas"] = n_replicas
                else:
                    if not list_or_dict[key]["n_replicas"] == n_replicas:
                        raise DimsNotMatchingError(
                            n_replicas,
                            list_or_dict[key]["n_replicas"],
                            "n_replicas of top level ({}) and {} ({}) do not match".format(
                                n_replicas, key, list_or_dict[key]["n_replicas"]
                            ),
                        )

            return list_or_dict

        set_par_list_or_dict(values, set_fn=partial(set_n_replicas))

        return values


Result = Dict[str, Any]


class MCMCResult(BaseConfig):

    _name: str = "mcmc_result"

    mcmc_config: Optional[MCMCConfig]
    results: Optional[List[Result]]
    acceptance_rate: Optional[float]
    obs_stats: Optional[Dict[str, Any]]
    sus_exact: Optional[float]
