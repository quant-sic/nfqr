import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, root_validator, validator

from nfqr.config import BaseConfig
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.normalizing_flows.flow import FlowConfig
from nfqr.normalizing_flows.loss.loss import LossConfig
from nfqr.target_systems import OBSERVABLE_REGISTRY, ActionConfig
from nfqr.train.scheduler import SchedulerConfig
from nfqr.utils import DimsNotMatchingError, create_logger, set_par_list_or_dict

logger = create_logger(__name__)

ConfigType = TypeVar("ConfigType", bound="LitModelConfig")


class TrainerConfig(BaseModel):

    batch_size: int
    train_num_batches: int
    val_num_batches: int
    max_epochs: int = 100
    accumulate_grad_batches: int = 1

    log_every_n_steps: int = 50

    learning_rate: float = 0.001
    auto_lr_find: bool = False

    loss_configs: List[LossConfig]
    loss_scheduler_config: SchedulerConfig = None

    scheduler_configs: Optional[List[SchedulerConfig]] = []

    n_iter_eval: int = 10
    batch_size_eval: int = 10000

    gradient_clip_val: Union[float, None] = 1.0
    gradient_clip_algorithm: str = "norm"
    track_grad_norm: int = 2

    optimizer: str = "Adam"
    lr_scheduler: Optional[Dict[str, Union[float, int, str]]] = {
        "type": "reduce_on_plateau"
    }

    stats_limits: List[int] = [-1]
    p_sampler_set_size:int = 250000


class LitModelConfig(BaseConfig):

    _name: str = "train_config"

    flow_config: FlowConfig
    action_config: ActionConfig
    dim: List[int]

    observables: List[str]
    trainer_configs: List[TrainerConfig]

    task_parameters: Union[List[str], None] = None
    initial_weights: Optional[Path] = None

    continue_model: Optional[Path]
    continue_beta: Optional[float]
    continuation_exp: Optional[Path]

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
                        "Len could not be evaluated for {}".format(
                            list_or_dict[key])
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
                list(num_pars_dict.values())[0] if list(
                    num_pars_dict.values()) else 1
            )

        return num_pars

    @classmethod
    def from_directory_for_task(
        cls: Type[ConfigType],
        directory: Union[str, Path],
        task_id,
        num_tasks=None,
        tune_config=None,
    ) -> ConfigType:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        num_pars_dict = {}

        def choose_task_par(key, list_or_dict, task_id):

            if key in raw_config["task_parameters"]:
                try:
                    num_pars_dict[key] = len(list_or_dict[key])
                except TypeError:
                    raise RuntimeError(
                        "Len could not be evaluated for {}".format(
                            list_or_dict[key])
                    )
                list_or_dict[key] = list_or_dict[key][task_id]

            return list_or_dict

        if raw_config["task_parameters"] is not None:
            raw_config = set_par_list_or_dict(
                raw_config, set_fn=partial(choose_task_par, task_id=task_id)
            )

        # check for inconsistencies in task array setup and config
        if not len(set(num_pars_dict.values())) <= 1:
            raise ValueError(
                f"Inconsistent number of tasks for parameters. {num_pars_dict}"
            )
        else:
            num_pars = (
                list(num_pars_dict.values())[0] if list(
                    num_pars_dict.values()) else 1
            )

            if num_tasks is not None and not num_pars == num_tasks:
                raise ValueError(
                    "Number of started tasks {} does not match number of tasks configured {}".format(
                        num_tasks, num_pars_dict
                    )
                )

        if tune_config is not None:

            def set_tuned_par(key, list_or_dict):
                if key in tune_config.keys():
                    list_or_dict[key] = tune_config[key]
                return list_or_dict

            raw_config = set_par_list_or_dict(raw_config, set_fn=set_tuned_par)

        return cls(**raw_config)

    @root_validator(pre=True)
    @classmethod
    def add_dims(cls, values):
        """
        Adds dims to sub configs.
        """

        dim = values["dim"]

        def set_dim(key, list_or_dict):

            if key in (
                "layer_chain_config",
                "base_dist_config",
                "trajectory_sampler_config",
                "specific_action_config"
            ):
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
    def add_action_config(cls, values):
        """
        Adds action_config to sub configs.
        """

        action_config = values["action_config"]

        def set_action_config(key, list_or_dict):

            if key in (
                "trajectory_sampler_config",
            ):
                if "action_config" not in list_or_dict[key]:

                    list_or_dict[key]["action_config"] = action_config

            return list_or_dict

        set_par_list_or_dict(values, set_fn=partial(set_action_config))

        return values

    @validator("trainer_configs", pre=True)
    @classmethod
    def trainer_configs_to_list(cls, _v):
        if not isinstance(_v, list):
            v = [_v]
        else:
            v = _v

        return v

    @validator("initial_weights", pre=True)
    @classmethod
    def initial_weights_add_exp_dir(cls, _v):

        if not isinstance(_v, Path):
            _v = Path(_v)

        v = EXPERIMENTS_DIR / _v

        return v
