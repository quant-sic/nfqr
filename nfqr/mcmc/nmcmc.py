import math

import torch
from numpy.random import rand

from nfqr.mcmc.base import MCMC


class NeuralMCMC(MCMC):
    def __init__(self, n_steps, model, target, observables_rec, trove_size):
        super(NeuralMCMC, self).__init__(n_steps)

        self.model = model
        self.target = target

        # set model to evaluation mode
        self.model.eval()

        self.trove_size = trove_size
        self.trove = 0
        self.previous_weight = 0.0

        self.observables_rec = observables_rec

    def step(self):

        with torch.no_grad():
            if self.n_current_steps % self.trove_size == 0:
                self._replentish_trove()

        idx = self.n_current_steps % self.trove_size
        weight_proposed = self.trove["weights"][idx]
        log_ratio = (weight_proposed - self.previous_weight).item()

        if log_ratio >= 0 or math.log(rand()) < log_ratio:
            self.n_accepted += 1
            self.previous_weight = weight_proposed
            self.current_config = self.trove["configs"][idx].unsqueeze(0)

        self.observables_rec.record_config(self.current_config)

    @property
    def acceptance_rate(self):
        return self.n_accepted/self.n_current_steps

    def initialize(self):

        with torch.no_grad():
            config, log_prob = self.model.sample_with_abs_log_det((1,))

        self.current_config = config
        self.previous_weight = self.target.log_prob(config) - log_prob

    def _replentish_trove(self):
        configs, log_probs = self.model.sample_with_abs_log_det((self.trove_size,))
        self.trove = {
            "configs": configs,
            "weights": self.target.log_prob(configs) - log_probs,
        }
