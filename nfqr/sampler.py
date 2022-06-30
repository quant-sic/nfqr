from abc import ABC, abstractmethod

from nfqr.recorder import ObservableRecorder
from nfqr.target_systems import OBSERVABLE_REGISTRY


class Sampler(ABC):
    def __init__(self, observables, out_dir, target_system, n_replicas) -> None:

        self._observables_rec = ObservableRecorder(
            observables={
                obs: OBSERVABLE_REGISTRY[target_system][obs]() for obs in observables
            },
            save_dir_path=out_dir,
            delete_existing_data=True,
            n_replicas=n_replicas,
        )

    @property
    def observables_rec(self):
        return self._observables_rec

    @abstractmethod
    def _evaluate_obs(self):
        raise NotImplementedError()

    def aggregate(self):

        stats = {}
        for obs in self.observables_rec.observables:
            stats[obs] = self._evaluate_obs(obs)

        return stats

    @abstractmethod
    def _stats(self):
        raise NotImplementedError()

    def get_stats(self):
        return self._stats
