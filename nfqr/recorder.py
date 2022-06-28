import io
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch


class ObservableRecorder(object):
    def __init__(
        self,
        observables: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
        save_dir_path: Path,
        delete_existing_data: bool = False,
    ) -> None:

        self.delete_existing_data = delete_existing_data
        self.observables = observables

        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

    @cached_property
    def observable_fstreams(self):
        streams = {}
        for name, path in self.observable_save_paths.items():
            if path.is_file() and self.delete_existing_data:
                os.remove(path)
            streams[name] = io.open(path, "ab")

        return streams

    @cached_property
    def log_weights_fstream(self):
        if self.log_weights_save_path.is_file() and self.delete_existing_data:
            os.remove(self.log_weights_save_path)

        return io.open(self.log_weights_save_path, "ab")

    @cached_property
    def log_weights_save_path(self):
        return self.save_dir_path / "log_weights"

    @cached_property
    def log_p_fstream(self):
        if self.log_p_save_path.is_file() and self.delete_existing_data:
            os.remove(self.log_p_save_path)

        return io.open(self.log_p_save_path, "ab")

    @cached_property
    def log_p_save_path(self):
        return self.save_dir_path / "log_p"

    @cached_property
    def observable_save_paths(self):

        observable_paths = {}
        for name in self.observables.keys():

            observable_paths[name] = self.save_dir_path / name

        return observable_paths

    def evaluate_observables(self, config: torch.Tensor):

        obs_values = {}

        for name, obs in self.observables.items():
            obs_values[name] = obs.evaluate(config)

        return obs_values

    def record_log_weight(self, log_weight,log_p=None):

        self.log_weights_fstream.write(
            log_weight.cpu().numpy().flatten().astype(np.float32).tobytes()
        )

        if log_p is not None:
            self.log_p_fstream.write(
                log_p.cpu().numpy().flatten().astype(np.float32).tobytes()
            )

    def record_obs(self, name, obs):
        self.observable_fstreams[name].write(
            obs.cpu().numpy().flatten().astype(np.float32).tobytes()
        )

    def record_config(self, config):

        obs_values = self.evaluate_observables(config)
        for name, value in obs_values.items():
            self.record_obs(name, value)

    def record_config_with_log_weight(self, config, log_weight,log_p=None):

        obs_values = self.evaluate_observables(config)
        for name, value in obs_values.items():
            self.record_obs(name, value)

        self.record_log_weight(log_weight,log_p)

    def flush_streams(self):

        if "observable_fstreams" in self.__dict__:
            for stream in self.observable_fstreams.values():
                stream.flush()

        if "log_weights_fstream" in self.__dict__:
            self.log_weights_fstream.flush()

        if "log_p_fstream" in self.__dict__:
            self.log_p_fstream.flush()

    def _load_file(self, path):
        self.flush_streams()

        with io.open(path, "rb") as file:
            file_tensor = torch.from_numpy(np.fromfile(file, dtype=np.float32))

        return file_tensor

    def __getitem__(self, name):

        if name == "log_weights":
            if self.log_weights_save_path.is_file():
                path = self.log_weights_save_path
            else:
                raise ValueError("No weights have been recorded")
        elif name == "log_p":
            if self.log_p_save_path.is_file():
                path = self.log_p_save_path
            else:
                raise ValueError("No p probs have been recorded")
        else:
            if name not in self.observable_save_paths:
                raise ValueError("Unknown name")
            else:
                path = self.observable_save_paths[name]

        return self._load_file(path)
