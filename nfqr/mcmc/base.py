from tqdm import tqdm

from nfqr.mcmc.stats import get_mcmc_statistics
from nfqr.sampler import Sampler
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class MCMC(Sampler):
    def __init__(
        self, n_steps, observables, target_system, out_dir, n_replicas,delete_existing_data=True
    ) -> None:
        super(MCMC, self).__init__(
            observables=observables,
            target_system=target_system,
            out_dir=out_dir,
            n_replicas=n_replicas,delete_existing_data=delete_existing_data
        )
        self.n_steps = n_steps
        self._config = None
        self.n_accepted = 0
        self.n_current_steps = 0

    def step(self):
        pass

    def initialize(self):
        pass

    @property
    def stats_limit(self):
        if not hasattr(self,"_stats_limit"):
            self._stats_limit = -1
        
        return self._stats_limit            

    @stats_limit.setter
    def stats_limit(self,v):
        self._stats_limit = v

    @property
    def current_config(self):
        return self._config

    @current_config.setter
    def current_config(self, new_current_config):
        self._config = new_current_config

    def __iter__(self):

        self.initialize()

        for _ in range(self.n_steps):
            self.step()
            self.n_current_steps += 1
            yield self.current_config

    def run(self):
        for _ in tqdm(self, total=self.n_steps, desc="Running MCMC"):
            pass

    @property
    def n_skipped(self):
        return 0

    @property
    def acceptance_rate(self):
        return 0

    @property
    def step_size(self):
        return 0

    @property
    def _stats(self):
        return {
            "acc_rate": self.acceptance_rate,
            "n_steps": self.n_current_steps,
            "obs_stats": self.aggregate(),
            "n_skipped": self.n_skipped,
            "step_size":self.step_size
        }

    def _evaluate_obs(self, obs):

        observable_data = self.observables_rec[obs][...,:self.stats_limit]
        prepared_observable_data = self.observables_rec.observables[obs].prepare(
            observable_data
        )

        stats = get_mcmc_statistics(prepared_observable_data)
        stats_postprocessed = self.observables_rec.observables[obs].postprocess(stats)

        return stats_postprocessed
