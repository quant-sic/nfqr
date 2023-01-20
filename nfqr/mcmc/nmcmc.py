import math

import torch
from numpy.random import rand

from nfqr.mcmc.base import MCMC
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class NeuralMCMC(MCMC):
    def __init__(
        self,
        n_steps,
        model,
        target,
        observables,
        out_dir,
        trove_size,
        target_system="qr",
        n_replicas: int = 1,
    ):
        super(NeuralMCMC, self).__init__(
            n_steps=n_steps,
            observables=observables,
            target_system=target_system,
            out_dir=out_dir,
            n_replicas=n_replicas,
        )

        self.model = model.double()
        self.target = target
        self.n_replicas = n_replicas

        # set model to evaluation mode
        self.model.eval()

        self.trove_size = trove_size
        self._trove = None
        self.previous_weight = 0.0
        self._n_skipped = 0

    def step(self):

        log_weight_of_proposed_config, proposed_config = self._get_next_tranche()
        log_ratio = (log_weight_of_proposed_config - self.previous_weight).item()

        if log_ratio >= 0 or math.log(rand()) < log_ratio:
            self.n_accepted += 1
            self.previous_weight = log_weight_of_proposed_config
            self.current_config = proposed_config.unsqueeze(0)

        self.observables_rec.record_config(self.current_config)

    @MCMC.n_skipped.getter
    def n_skipped(self):
        return self._n_skipped

    def _get_next_tranche(self):

        if self._trove is None or self.idx_in_trove == 0:
            self._replenish_trove()

        # gets next idx in trove which should not be skipped
        while (self._trove["skip"][self.idx_in_trove]).item():
            self._n_skipped += 1

            if self.idx_in_trove == 0:
                self._replenish_trove()

        return (
            self._trove["log_weights"][self.idx_in_trove],
            self._trove["configs"][self.idx_in_trove],
        )

    @property
    def idx_in_trove(self):
        return (self.n_current_steps + self._n_skipped) % self.trove_size

    @property
    def acceptance_rate(self):
        return self.n_accepted / self.n_current_steps

    def initialize(self):

        with torch.no_grad():
            config, log_prob = self.model.sample_with_abs_log_det((self.n_replicas,))

        self.current_config = config
        self.previous_weight = self.target.log_prob(config) - log_prob

    def _replenish_trove(self):

        with torch.no_grad():
            configs, log_probs = self.model.sample_with_abs_log_det((self.trove_size,))

        log_weights = self.target.log_prob(configs) - log_probs

        self._trove = {
            "configs": configs,
            "log_weights": log_weights,
            "skip": torch.isnan(log_weights) | torch.isinf(log_weights),
        }


class NeuralMCMCParallel(MCMC):
    def __init__(
        self,
        n_steps,
        model,
        target,
        observables,
        out_dir,
        trove_size,
        target_system="qr",
        n_replicas: int = 1,
        n_record_skips: int = 1,
    ):
        super(NeuralMCMCParallel, self).__init__(
            n_steps=n_steps,
            observables=observables,
            target_system=target_system,
            out_dir=out_dir,
            n_replicas=n_replicas,
        )

        self.model = model.double()
        self.target = target
        self.n_replicas = n_replicas
        self.n_record_skips = n_record_skips

        # set model to evaluation mode
        self.model.eval()

        self.trove_size = trove_size
        self._trove = None
        self.previous_weight = 0.0
        self._n_skipped = 0

        self._n_accepted = torch.zeros(n_replicas, dtype=torch.long)

    def step(self):

        log_weight_of_proposed_config, proposed_config = self._get_next_tranche()
        log_ratio = log_weight_of_proposed_config - self.previous_weight

        accept_mask = (log_ratio >= 0) | (math.log(rand()) < log_ratio)

        self.n_accepted += accept_mask

        self.previous_weight = (accept_mask * log_weight_of_proposed_config) + (
            (~accept_mask) * self.previous_weight
        )
        self.current_config = (accept_mask[:, None] * proposed_config) + (
            (~accept_mask[:, None]) * self.current_config
        )

        if self.n_current_steps % self.n_record_skips == 0:
            self.observables_rec.record_config(self.current_config)

    @MCMC.n_skipped.getter
    def n_skipped(self):
        return self._n_skipped

    def _get_next_tranche(self):

        if self._trove is None or self.idx_in_trove == 0:
            self._replenish_trove()

        # gets next idx in trove which should not be skipped
        while (self._trove["skip"][:, self.idx_in_trove].any()).item():
            self._n_skipped += 1

            if self.idx_in_trove == 0:
                self._replenish_trove()

        return (
            self._trove["log_weights"][:, self.idx_in_trove],
            self._trove["configs"][:, self.idx_in_trove],
        )

    @property
    def actual_trove_size(self):
        return self.trove_size_per_replicum * self.n_replicas

    @property
    def trove_size_per_replicum(self):
        return self.trove_size // self.n_replicas

    @property
    def idx_in_trove(self):
        return (self.n_current_steps + self._n_skipped) % self.trove_size_per_replicum

    @property
    def acceptance_rate(self):
        return self._n_accepted.to(torch.float32).mean() / self.n_current_steps

    def initialize(self):

        with torch.no_grad():
            config, log_prob = self.model.sample_with_abs_log_det((self.n_replicas,))

        self.current_config = config
        self.previous_weight = self.target.log_prob(config) - log_prob

    def _replenish_trove(self):

        with torch.no_grad():
            configs, log_probs = self.model.sample_with_abs_log_det(
                (self.actual_trove_size,)
            )

        configs = configs.reshape(self.n_replicas, -1, configs.shape[-1])
        log_probs = log_probs.reshape(self.n_replicas, -1)

        log_weights = self.target.log_prob(configs) - log_probs

        self._trove = {
            "configs": configs,
            "log_weights": log_weights,
            "skip": torch.isnan(log_weights) | torch.isinf(log_weights),
        }
