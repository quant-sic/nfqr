from typing import Literal

import torch
from tqdm import tqdm

from nfqr.nip.stats import (
    calc_entropy,
    calc_free_energy,
    calc_std_entropy,
    calc_std_free_energy,
    get_impsamp_statistics,
)
from nfqr.sampler import Sampler
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class NeuralImportanceSampler(Sampler):
    def __init__(
        self,
        model,
        observables,
        target,
        n_iter,
        out_dir,
        target_system="qr",
        batch_size=2000,
        mode: Literal["q", "p"] = "q",
        sampler=None,
    ):
        super(NeuralImportanceSampler, self).__init__(
            observables=observables, target_system=target_system, out_dir=out_dir,n_replicas=1
        )

        self.batch_size = batch_size
        self.model = model.double()
        self.target = target
        self.n_iter = n_iter

        # set model to evaluation mode
        self.model.eval()

        self.mode = mode

        if mode == "q":
            self.step = self.step_q
        elif mode == "p":
            self.step = self.step_p
            if sampler is None:
                raise ValueError("In p mode, sampler must not be None")
            else:
                self.sampler = sampler
                if not self.sampler.batch_size == self.batch_size:
                    logger.info(
                        f"Sampler yields batch_size {sampler.batch_size}, which is different than set batch_size {self.batch_size}"
                    )
        else:
            raise ValueError("Unknown mode")

    @property
    def stats_limit(self):
        if not hasattr(self,"_stats_limit"):
            self._stats_limit = -1
        
        return self._stats_limit            

    @stats_limit.setter
    def stats_limit(self,v):
        self._stats_limit = v
        
    def run(self):

        for _ in tqdm(range(self.n_iter), desc="Running NIP"):
            self.step()

    @torch.no_grad()
    def step_p(self):

        x_samples = self.sampler.sample(next(self.model.parameters()).device).detach()
        x_samples = x_samples.to(self.model.parameters().__next__().dtype)
        
        log_p = self.target.log_prob(x_samples)
        log_weights = log_p - self.model.log_prob(x_samples)

        self.observables_rec.record_config_with_log_weight(
            x_samples, log_weights, log_p=log_p
        )

    @torch.no_grad()
    def step_q(self):

        x_samples, log_q_x = self.model.sample_with_abs_log_det((self.batch_size,))
        log_p = self.target.log_prob(x_samples)

        log_weights = log_p - log_q_x

        self.observables_rec.record_config_with_log_weight(
            x_samples, log_weights, log_p=log_p
        )

    @property
    def unnormalized_log_weights(self):
        return self.observables_rec["log_weights"][...,:self.stats_limit]

    @property
    def log_p(self):
        return self.observables_rec["log_p"][...,:self.stats_limit]

    def _evaluate_obs(self, obs):

        observable_data = self.observables_rec[obs][...,:self.stats_limit]
        prepared_observable_data = self.observables_rec.observables[obs].prepare(
            observable_data
        )

        config_log_weights_unnormalized = self.observables_rec["log_weights"][...,:self.stats_limit]

        assert config_log_weights_unnormalized.shape == observable_data.shape

        stats = get_impsamp_statistics(
            prepared_observable_data, config_log_weights_unnormalized
        )

        stats_postprocessed = self.observables_rec.observables[obs].postprocess(stats)

        return stats_postprocessed

    @property
    def _stats(self):
        return {
            "obs_stats": self.aggregate(),
            "entropy": calc_entropy(
                self.unnormalized_log_weights,
                self.log_p,
                self.model.base_distribution.dim,
            ),
            "std_entropy": calc_std_entropy(
                self.unnormalized_log_weights,
                self.log_p,
                self.model.base_distribution.dim,
            ),
            "free_energy": calc_free_energy(
                self.unnormalized_log_weights, self.model.base_distribution.dim
            ),
            "std_free_energy": calc_std_free_energy(
                self.unnormalized_log_weights, self.model.base_distribution.dim
            ),
        }
