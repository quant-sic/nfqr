import json
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, root_validator

from nfqr.config import BaseConfig
from nfqr.data.config import PSamplerConfig
from nfqr.normalizing_flows.flow import FlowConfig
from nfqr.normalizing_flows.loss.loss import LossConfig
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig
from nfqr.train.scheduler import (
    BetaSchedulerConfig,
    LossSchedulerConfig,
    SchedulerConfig,
)
from nfqr.utils import create_logger, set_par_list_or_dict

logger = create_logger(__name__)

ConfigType = TypeVar("ConfigType", bound="TrainConfig")





class TrainerConfig(BaseModel):

    batch_size: int
    train_num_batches: int
    val_num_batches: int
    max_epochs: int = 100
    accumulate_grad_batches: int = 1

    log_every_n_steps: int = 50

    task_parameters: Union[List[str], None] = None
    
    learning_rate:float=0.001
    auto_lr_find:bool=False

    loss_configs: List[LossConfig]
    loss_scheduler_config : SchedulerConfig =None

    scheduler_configs: Optional[List[SchedulerConfig]] = []

    n_iter_eval: int = 5
    batch_size_eval: int = 10000
    




class LitModelConfig(BaseConfig):

    _name: str = "train_config"

    flow_config: FlowConfig
    action_config: ActionConfig
    dim: List[int]

    observables: List[OBSERVABLE_REGISTRY.enum]
    trainer_config: TrainerConfig


    @classmethod
    def get_num_tasks(cls:Type[ConfigType], directory: Union[str, Path]) -> int:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        num_pars_dict = {}


        def fill_num_pars_dict(key, list_or_dict):

            if key in raw_config["trainer_config"]["task_parameters"]:
                try:
                    num_pars_dict[key] = len(list_or_dict[key])
                except TypeError:
                    raise RuntimeError(
                        "Len could not be evaluated for {}".format(list_or_dict[key])
                    )
            return list_or_dict

        if raw_config["trainer_config"]["task_parameters"] is not None:
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
        cls: Type[ConfigType], directory: Union[str, Path], task_id, num_tasks
    ) -> ConfigType:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        num_pars_dict = {}

        def choose_task_par(key, list_or_dict, task_id):

            if key in raw_config["trainer_config"]["task_parameters"]:
                try:
                    num_pars_dict[key] = len(list_or_dict[key])
                except TypeError:
                    raise RuntimeError(
                        "Len could not be evaluated for {}".format(list_or_dict[key])
                    )
                list_or_dict[key] = list_or_dict[key][task_id]

            return list_or_dict

        if raw_config["trainer_config"]["task_parameters"] is not None:
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
                list(num_pars_dict.values())[0] if list(num_pars_dict.values()) else 1
            )

            if not num_pars == num_tasks:
                raise ValueError(
                    "Number of started tasks {} does not match number of tasks configured {}".format(
                        num_tasks, num_pars_dict
                    )
                )

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
