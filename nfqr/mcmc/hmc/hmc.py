from typing import List, Tuple

import numpy as np
import numpyro
import torch
from jax import random as jax_random
from tqdm.autonotebook import tqdm

from nfqr.mcmc.base import MCMC
from nfqr.mcmc.hmc.hmc_cpp import hmc_cpp
from nfqr.registry import StrRegistry
from nfqr.target_systems import ActionConfig
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


HMC_REGISTRY = StrRegistry("hmc")


@HMC_REGISTRY.register("leapfrog")
class HMC(MCMC):
    def __init__(
        self,
        n_steps: int,
        dim: List[int],
        action_config: ActionConfig,
        n_burnin_steps: int,
        observables,
        out_dir,
        target_system: str = "qr",
        n_traj_steps=20,
        step_size=0.01,
        autotune_step=True,
        alg="cpp_batch",
        batch_size=10000,
        n_samples_at_a_time=10000,
        action="qr",
        **kwargs,
    ) -> None:
        super(HMC, self).__init__(
            n_steps=n_steps,
            observables=observables,
            target_system=target_system,
            out_dir=out_dir,
        )

        self.dim = dim
        self.batch_size = batch_size
        self.n_burnin_steps = n_burnin_steps
        self.n_steps = n_steps
        self.step_size = step_size
        self.n_traj_steps = n_traj_steps
        self.n_samples_at_a_time = n_samples_at_a_time

        # py_action = ACTION_REGISTRY[target_system][action](**dict(action_config))

        self.alg = alg
        if "cpp" in alg:
            if not target_system == "qr" or not action == "qr":
                raise ValueError("For Cpp algorithm currently only qr is supported")

            if len(observables) > 1:
                raise ValueError(
                    "For Cpp algorithm currently only 1 observable allowed"
                )

            if "Chi_t" in observables:
                cpp_obs = hmc_cpp.TopologicalSusceptibility()
                self.observable = "Chi_t"
            else:
                raise ValueError("Unknown Observable")

            if action == "qr":
                cpp_action = hmc_cpp.QR(action_config.beta)
            else:
                raise ValueError("Unknown Action")

            if alg == "cpp_batch":
                self.hmc = hmc_cpp.HMC_Batch(cpp_obs, cpp_action, dim[0], batch_size)
            elif alg == "cpp_single":
                self.hmc = hmc_cpp.HMC_Single_Config(cpp_obs, cpp_action, dim[0])
            else:
                raise ValueError("Unknown Algorithm")
        elif "python" in alg:
            raise NotImplementedError()
        else:
            raise ValueError("Unknown Algorithm")

        if autotune_step:
            self.autotune_step_size(0.8)

        self._trove = None

    @property
    def acceptance_rate(self):
        return self.hmc.acceptance_rate

    def initialize(self):

        logger.info("Initializing HMC")
        self.hmc.initialize()

        logger.info(f"Burnin steps :{self.n_burnin_steps}")
        self.hmc.burnin(self.n_burnin_steps, self.n_traj_steps, self.step_size)

        return self.hmc.current_config

    def step(self):

        step_in_trove = self.n_current_steps % self.n_samples_at_a_time
        if (self._trove is None) or (step_in_trove == 0):
            self.hmc.reset_expectation_values()
            self.hmc.advance(
                self.n_samples_at_a_time, self.n_traj_steps, self.step_size
            )

            if isinstance(self.hmc.expectation_values[0], torch.Tensor):
                self._trove = torch.stack(self.hmc.expectation_values, dim=-1)
            else:
                self._trove = torch.Tensor(self.hmc.expectation_values)

        self.observables_rec.record_obs(
            self.observable, self._trove[..., step_in_trove]
        )

    def autotune_step_size(self, desired_acceptance_percentage):

        n_autotune_samples = 1000
        tolerance = 0.05  # Tolerance
        step_size_original = 0.01
        step_size_min = 0.01 * step_size_original
        step_size_max = 100 * step_size_original
        converged = False
        tune_steps = 100

        pbar = tqdm(range(tune_steps))
        for _ in pbar:

            self.hmc.initialize()
            self.step_size = 0.5 * (step_size_min + step_size_max)

            self.hmc.advance(
                n_steps=n_autotune_samples,
                n_traj_steps=self.n_traj_steps,
                step_size=self.step_size,
            )

            acceptance_rate = (
                self.hmc.acceptance_rate.mean()
                if isinstance(self.hmc.acceptance_rate, torch.Tensor)
                else self.hmc.acceptance_rate
            )
            if acceptance_rate > desired_acceptance_percentage:
                step_size_min = self.step_size
            else:
                step_size_max = self.step_size

            if abs(acceptance_rate - desired_acceptance_percentage) < tolerance:
                converged = True
                break

            pbar.set_description(
                f"step_size: {self.step_size}, Acceptance Rate {acceptance_rate}"
            )

        if not converged:
            self.step_size = step_size_original


class HMC_NUMPYRO(MCMC):
    def __init__(self, dim, target, n_burin_stpes, n_steps, **kwargs) -> None:
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
        for _sample in samples:
            yield _sample

    def step(self):
        self.current_config = self.samples_iter.__next__()
