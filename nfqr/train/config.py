import json
from pathlib import Path
from typing import List, Literal, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, root_validator

from nfqr.config import BaseConfig
from nfqr.normalizing_flows.flow import FlowConfig
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY
from nfqr.target_systems.config import ActionConfig

ConfigType = TypeVar("ConfigType", bound="TrainConfig")


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
        cls: Type[ConfigType], directory: Union[str, Path], task_id
    ) -> ConfigType:
        """Load config from json with task id."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)

        def set_task_par(_dict):
            for key, value in _dict.items():
                if isinstance(value, dict):
                    _dict[key] = set_task_par(value)

                if key in raw_config["trainer_config"]["task_parameters"]:
                    _dict[key] = _dict[key][task_id]

            return _dict

        if raw_config["trainer_config"]["task_parameters"] is not None:
            raw_config = set_task_par(raw_config)

        return cls(**raw_config)

    @root_validator(pre=True)
    @classmethod
    def add_dims(cls, values):

        dim = values["dim"]
        if "dim" not in values["flow_config"]["layer_chain_config"]:
            values["flow_config"]["layer_chain_config"]["dim"] = dim
        else:
            assert dim == values["flow_config"]["layer_chain_config"]["dim"]

        if "dim" not in values["flow_config"]["base_dist_config"]:
            values["flow_config"]["base_dist_config"]["dim"] = dim
        else:
            assert dim == values["flow_config"]["base_dist_config"]["dim"]

        return values
