"""Base config for dataset-/task-specific training configs."""

import json
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

from pydantic import BaseModel

ConfigType = TypeVar("ConfigType", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for model, data and run configs."""

    _name: str = "base_config"

    def save(self, directory: Path) -> None:
        """Saves the config to specified directory."""
        directory.mkdir(parents=True, exist_ok=True)
        with open(self._config_path(directory), "w") as f:
            f.write(self.json(indent=2))

    def update_from_dict(self, param_dict: Dict[str, Any]) -> None:
        """Updates config from a parameter dict."""
        for k, v in param_dict.items():
            if k in self.__dict__:
                self.__dict__[k] = v

    @classmethod
    def from_directory(
        cls: Type[ConfigType], directory: Union[str, Path]
    ) -> ConfigType:
        """Load config from json."""
        with open(str(cls._config_path(Path(directory)))) as f:
            raw_config = json.load(f)
        return cls(**raw_config)

    @classmethod
    def _config_path(cls, directory: Path) -> Path:
        return directory.joinpath(cls._name + ".json")
