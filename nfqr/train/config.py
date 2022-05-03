import json
from pathlib import Path
from tkinter import N
from typing import List, Literal, Tuple, Type, TypeVar, Union

from pydantic import root_validator

from nfqr.config import BaseConfig
from nfqr.normalizing_flows.flow.config import FlowConfig
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY
from nfqr.target_systems.config import ActionConfig

ConfigType = TypeVar("ConfigType", bound="TrainConfig")


class TrainConfig(BaseConfig):

    _name: str = "train_config"

    flow_config: FlowConfig
    target_system: ACTION_REGISTRY.enum
    action: ACTION_REGISTRY.enum
    observables: List[OBSERVABLE_REGISTRY.enum]
    train_setup: Literal["reverse"] = "reverse"

    action_config: ActionConfig

    dim: Tuple[int]

    batch_size: int
    num_batches: int
    max_epochs: int = 10
    log_every_n_steps: int = 50
    task_parameters: Union[List[str], None] = None

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

        return cls(**raw_config)

    @root_validator()
    @classmethod
    def check_matching_dims(cls, values):
        print(values, type(values))

        dims = (
            values["dim"],
            values["flow_config"].layer_chain_config.dim,
            values["flow_config"].base_dist_config.dim,
        )
        assert len(set(dims)) == 1

        return values
