import json
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import List, Literal, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, root_validator

from nfqr.config import BaseConfig
from nfqr.normalizing_flows.flow import FlowConfig
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig

ConfigType = TypeVar("ConfigType", bound="TrainConfig")


class DimsNotMatchingError(Exception):
    def __init__(self, a, b, message) -> None:
        self.a = a
        self.b = b
        self.message = message
        super().__init__(message)


class TrainerConfig(BaseModel):

    batch_size: int
    num_batches: int
    max_epochs: int = 100
    log_every_n_steps: int = 50
    task_parameters: Union[List[str], None] = None
    accumulate_grad_batches: int = 1


class TrainConfig(BaseConfig):

    _name: str = "train_config"

    flow_config: FlowConfig
    target_system: ACTION_REGISTRY.enum
    action: ACTION_REGISTRY.enum
    observables: List[OBSERVABLE_REGISTRY.enum]
    train_setup: Literal["reverse"] = "reverse"

    action_config: ActionConfig

    dim: Tuple[int]
    trainer_config: TrainerConfig

    @classmethod
    def from_directory_for_task(
        cls: Type[ConfigType], directory: Union[str, Path], task_id, num_tasks
    ) -> ConfigType:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        # recursively set task parameters. Might be problemati bc mutable object is read and written to in parallel
        num_pars_dict = {}

        def set_task_par(_dict):
            for key, value in _dict.items():
                if isinstance(value, dict):
                    _dict[key] = set_task_par(value)

                if key in raw_config["trainer_config"]["task_parameters"]:
                    num_pars_dict[key] = len(_dict[key])
                    _dict[key] = _dict[key][task_id]

            return _dict

        if raw_config["trainer_config"]["task_parameters"] is not None:
            raw_config = set_task_par(raw_config)

        # check for inconsistencies in task array setup and config
        if not len(set(num_pars_dict.values())) == 1:
            raise ValueError(
                f"Inconsistent number of tasks for parameters. {num_pars_dict}"
            )
        else:
            num_pars = list(num_pars_dict.values())[0]

            if not num_pars == num_tasks:
                raise ValueError(
                    "Number of started tasks {} does not match number of tasks configured {}".format(
                        num_tasks, len(raw_config["trainer_config"]["task_parameters"])
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
        if "dim" not in values["flow_config"]["layer_chain_config"]:
            values["flow_config"]["layer_chain_config"]["dim"] = dim
        else:
            if not dim == values["flow_config"]["layer_chain_config"]["dim"]:
                raise DimsNotMatchingError(
                    dim,
                    values["flow_config"]["layer_chain_config"]["dim"],
                    "Dim of top level ({}) and flow_config->layer_chain_config->dim ({}) do not match".format(
                        dim, values["flow_config"]["layer_chain_config"]["dim"]
                    ),
                )

        if "dim" not in values["flow_config"]["base_dist_config"]:
            values["flow_config"]["base_dist_config"]["dim"] = dim
        else:
            if not dim == values["flow_config"]["base_dist_config"]["dim"]:
                raise DimsNotMatchingError(
                    dim,
                    values["flow_config"]["base_dist_config"]["dim"],
                    "Dim of top level ({}) and flow_config->base_dist_config->dim ({}) do not match".format(
                        dim, values["flow_config"]["base_dist_config"]["dim"]
                    ),
                )

        return values
