"""
Code for observable generation and storage
"""
import io
from functools import cached_property

import numpy as np
import torch

from nfqr.nip.nip import calc_imp_weights
from nfqr.stats import get_iid_statistics, get_impsamp_statistics, get_mcmc_statistics


class Observable(object):
    def __init__(self) -> None:
        pass

    def prepare(self, history):
        return history

    def postprocess(self, value):
        return value


class ObservableRecorder(object):
    def __init__(
        self, observables, sampler, save_dir_path, stats_function=get_iid_statistics
    ) -> None:

        self.observables = observables
        self.sampler = sampler

        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(exist_ok=True)

        self.stats_function = stats_function

    @cached_property
    def observable_fstreams(self):
        streams = {}
        for name, path in self.observable_save_paths.items():

            streams[name] = io.open(path, "ab")

        return streams

    @cached_property
    def log_weights_fstream(self):
        return io.open(self.log_weights_save_path, "ab")

    @cached_property
    def log_weights_save_path(self):
        return self.save_dir_path / "log_weights"

    @cached_property
    def observable_save_paths(self):

        observable_paths = {}
        for name in self.observables.keys():

            observable_paths[name] = self.save_dir_path / name

        return observable_paths

    def record(self, config, log_weight=None):

        if log_weight is not None:
            self.log_weights_fstream.write(
                log_weight.cpu().numpy().flatten().astype(np.float32).tobytes()
            )

        for name, observable in self.observables.items():

            obs_values = observable.evaluate(config)

            self.observable_fstreams[name].write(
                obs_values.cpu().numpy().flatten().astype(np.float32).tobytes()
            )

    def record_sampler(self):

        for sampler_out in self.sampler:
            if isinstance(sampler_out, tuple):
                self.record(*sampler_out)
            else:
                self.record(sampler_out)

        if hasattr(self, "observable_fstreams"):
            for stream in self.observable_fstreams.values():
                stream.flush()
                stream.close()

        if hasattr(self, "log_weights_fstream"):
            self.log_weights_fstream.flush()

    def _load_file(self, path):
        with io.open(path, "rb") as file:
            file_tensor = torch.from_numpy(np.fromfile(file, dtype=np.float32))

        return file_tensor

    def __getitem__(self, name):

        if name == "log_weights":
            if self.log_weights_save_path.is_file():
                path = self.log_weights_save_path
            else:
                raise ValueError("No weights have been recorded")

        else:
            if name not in self.observable_save_paths:
                raise ValueError("Unknown name")
            else:
                path = self.observable_save_paths[name]

        return self._load_file(path)

    def load_imp_weights(self):
        return calc_imp_weights(self["log_weights"])

    def _evaluate_obs(self, observable):

        observable_data = self[observable]
        prepared_observable_data = self.observables[observable].prepare(observable_data)

        if self.stats_function == get_impsamp_statistics:
            config_weights_unnormalized = self.load_imp_weights()

            assert len(config_weights_unnormalized) == len(observable_data)

            stats = self.stats_function(
                prepared_observable_data, config_weights_unnormalized
            )

        else:
            stats = self.stats_function(prepared_observable_data)

        stats_postprocessed = self.observables[observable].postprocess(stats)

        return stats_postprocessed

    def aggregate(self):

        stats = {}
        for obs in self.observables.keys():
            stats[obs] = self._evaluate_obs(obs)

        return stats
