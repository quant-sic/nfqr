import numpy as np
import numpyro
import torch
from jax import random as jax_random
from pyro import sample

from nfqr.mcmc.base import MCMC


class HMC(MCMC):
    def __init__(self, model, target, warmup, n_steps) -> None:
        super().__init__(n_steps=n_steps)

        self.dim = model.base_distribution.dim
        self.target = target
        self._potential_fn = target.dist.action.evaluate_jnp

        hmc = numpyro.infer.HMC(potential_fn=self._potential_fn)
        self.mcmc = numpyro.infer.MCMC(hmc, num_warmup=warmup, num_samples=n_steps)

        self.rng_key = jax_random.PRNGKey(10)

    def initialize(self):
        self.init_params = np.random.random(self.dim)
        self.mcmc.run(self.rng_key, init_params=self.init_params)
        samples = torch.from_numpy(np.array(self.mcmc.get_samples()))
        self.samples_iter = self._samples_iter(samples=samples).__iter__()

    def _samples_iter(self, samples):
        for sample in samples:
            yield sample

    def step(self):
        self.current_config = self.samples_iter.__next__()
