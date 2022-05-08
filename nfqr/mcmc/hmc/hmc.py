from multiprocessing.sharedctypes import Value

import numpy as np
import numpyro
import torch
from jax import random as jax_random
from pyro import sample
from torch.utils import cpp_extension
from tqdm.autonotebook import tqdm

from nfqr.globals import REPO_ROOT
from nfqr.mcmc.base import MCMC
from nfqr.target_systems.rotor.rotor import QuantumRotor, TopologicalSusceptibility

hmc_cpp = cpp_extension.load(
    name="hmc_cpp",
    sources=[
        REPO_ROOT / "nfqr/target_systems/rotor/rotor.cpp",
        REPO_ROOT / "nfqr/mcmc/hmc/hmc.cpp",
        REPO_ROOT / "nfqr/mcmc/hmc/hmc_binding.cpp",
    ],
    extra_include_paths=[
        str(REPO_ROOT / "nfqr/target_systems"),
        str(REPO_ROOT / "nfqr/target_systems/rotor"),
    ],
)


class HMC_NUMPYRO(MCMC):
    def __init__(self, dim, target, n_burin_stpes, n_steps) -> None:
        super().__init__(n_steps=n_steps)

        self.dim = dim
        self.target = target
        self._potential_fn = target.dist.action.evaluate_jnp

        hmc = numpyro.infer.HMC(potential_fn=self._potential_fn)
        self.mcmc = numpyro.infer.MCMC(
            hmc, num_warmup=n_burin_stpes, num_samples=n_steps
        )

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


class HMC_CPP(MCMC):
    def __init__(
        self,
        n_steps,
        dim,
        batch_size,
        target,
        n_burnin_steps,
        step_size=0.01,
        autotune_step=True,
    ) -> None:
        super(HMC_CPP, self).__init__(n_steps=n_steps)

        self.hmc_cpp = hmc_cpp.HMC()
        self.dim = dim
        self.batch_size = batch_size

        self.n_burnin_steps = n_burnin_steps
        self.n_steps = n_steps

        self.step_size = step_size
        self.n_traj_steps = 100

        if isinstance(target.dist.action, QuantumRotor):
            self.action = hmc_cpp.QR(target.dist.action.beta)
        else:
            raise ValueError("Unknown Action")

        if autotune_step:
            self.autotune_step_size(0.8)

        # cpp_hmc.current_config()

        # sus = cpp_hmc.advance(
        #     0.01, 100, 10000, hmc_cpp.QR(1.0), [hmc_cpp.TopologicalSusceptibility()]
        # )
        # cpp_hmc.n_accepted()

    def initialize(self):

        input = torch.rand(1, self.dim, dtype=torch.float32)
        self.hmc_cpp.initialize(input)

        self.hmc_cpp.advance(
            self.step_size,
            self.n_traj_steps,
            self.n_burnin_steps,
            self.action,
            [],
            False,
        )

        return self.hmc_cpp.current_config()

    def step(self):
        return self.hmc_cpp.advance(
            self.step_size,
            self.n_traj_steps,
            self.batch_size,
            self.action,
            [hmc_cpp.TopologicalSusceptibility()],
            True,
        )

    def autotune_step_size(self, desired_acceptance_percentage):

        n_autotune_samples = 1000
        tolerance = 0.03  # Tolerance
        step_size_original = 0.01
        step_size_min = 0.01 * step_size_original
        step_size_max = 100 * step_size_original
        converged = False
        tune_steps = 100

        pbar = tqdm(range(tune_steps))
        for _ in pbar:

            self.initialize()
            self.step_size = 0.5 * (step_size_min + step_size_max)

            self.hmc_cpp.advance(
                self.step_size,
                self.n_traj_steps,
                n_autotune_samples,
                self.action,
                [],
                True,
            )

            acc_perc = self.hmc_cpp.n_accepted() / n_autotune_samples

            if acc_perc.mean() > desired_acceptance_percentage:
                step_size_min = self.step_size
            else:
                step_size_max = self.step_size

            if abs(acc_perc.mean() - desired_acceptance_percentage) < tolerance:
                converged = True
                break

            pbar.set_description(
                f"step_size: {self.step_size}, Acceptance Rate {acc_perc.mean()}"
            )

        if not converged:
            self.step_size = step_size_original
