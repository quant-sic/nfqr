from tqdm.autonotebook import tqdm

from nfqr.mcmc.stats import get_mcmc_statistics
from nfqr.sampler import Sampler
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class MCMC(Sampler):
    def __init__(self, n_steps, observables, target_system, out_dir) -> None:
        super(MCMC, self).__init__(
            observables=observables, target_system=target_system, out_dir=out_dir
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
        for _ in tqdm(self,total=self.n_steps, desc="Running MCMC"):
            pass

    @property
    def n_skipped(self):
        return 0

    @property
    def _stats(self):
        return {
            "acc_rate": self.acceptance_rate,
            "n_steps": self.n_current_steps,
            "obs_stats": self.aggregate(),
            "n_skipped": self.n_skipped
        }

    def _evaluate_obs(self, obs):

        observable_data = self.observables_rec[obs]
        prepared_observable_data = self.observables_rec.observables[obs].prepare(
            observable_data
        )

        stats = get_mcmc_statistics(prepared_observable_data)
        stats_postprocessed = self.observables_rec.observables[obs].postprocess(stats)

        return stats_postprocessed
