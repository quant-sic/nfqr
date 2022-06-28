from typing import List, Union

import numpy as np
import numpyro
import torch
from jax import random as jax_random
from tqdm import tqdm

from nfqr.mcmc.base import MCMC
from nfqr.mcmc.hmc.hmc_cpp import hmc_cpp
from nfqr.mcmc.initial_config import InitialConfigSampler
from nfqr.registry import StrRegistry
from nfqr.target_systems import ACTION_REGISTRY, ActionConfig
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
        n_traj_steps=20,
        step_size=0.01,
        autotune_step=True,
        hmc_engine="cpp_batch",
        batch_size=10000,
        n_samples_at_a_time=10000,
        initial_config_sampler_config=None,
        **kwargs,
    ) -> None:
        super(HMC, self).__init__(
            n_steps=n_steps,
            observables=observables,
            target_system=action_config.target_system,
            out_dir=out_dir,
        )

        self.dim = dim
        self.batch_size = batch_size
        self.n_burnin_steps = n_burnin_steps
        self.n_steps = n_steps
        self.initial_step_size = step_size
        self.n_traj_steps = n_traj_steps
        self.n_samples_at_a_time = n_samples_at_a_time

        self.target_system = action_config.target_system
        self.action = ACTION_REGISTRY[action_config.target_system][action_config.action_type](**dict(action_config.specific_action_config))

        self.hmc_engine = hmc_engine
        if "cpp" in hmc_engine:

            self.step = self._step_cpp
            self._trove = None

            if not action_config.target_system == "qr" or not action_config.action_type == "qr":
                raise ValueError("For Cpp hmc_engine currently only qr is supported")

            if len(observables) > 1:
                raise ValueError(
                    "For Cpp hmc_engine currently only 1 observable allowed"
                )

            if "Chi_t" in observables:
                cpp_obs = hmc_cpp.TopologicalSusceptibility()
                self.observable = "Chi_t"
            else:
                raise ValueError("Unknown Observable")

            if action_config.action_type == "qr":
                cpp_action = hmc_cpp.QR(action_config.beta)
            else:
                raise ValueError("Unknown Action")

            if hmc_engine == "cpp_batch":
                self.hmc = hmc_cpp.HMC_Batch(cpp_obs, cpp_action, dim[0], batch_size)
            elif hmc_engine == "cpp_single":
                self.hmc = hmc_cpp.HMC_Single_Config(cpp_obs, cpp_action, dim[0])
            else:
                raise ValueError("Unknown cpp hmc_engine")

        elif "python" in hmc_engine:
            self.hmc = HMC_PYTHON(action=self.action, dim=self.dim)
            self.step = self._step_python

        else:
            raise ValueError("Unknown hmc_engine")

        self.initial_config_sampler = InitialConfigSampler(
            **dict(initial_config_sampler_config)
        )

        self.autotune_step = autotune_step

    @property
    def step_size(self):
        if not hasattr(self, "_step_size") and self.autotune_step:
            self.autotune_step_size(0.8)

        else:
            self._step_size = self.initial_step_size

        return self._step_size

    @property
    def data_specs(self):
        return {
            "dim": self.dim,
            "beta": self.action.beta,
            "n_burnin_steps": self.n_burnin_steps,
            "n_traj_steps": self.n_traj_steps,
            "target_system": self.target_system,
            "data_sampler": "hmc",
        }

    @property
    def acceptance_rate(self):
        return self.hmc.acceptance_rate

    def initialize(self, burn_in=True, log=True):

        if log:
            logger.info("Initializing HMC")
        if "python" in self.hmc_engine:
            self.hmc.initialize(self.initial_config_sampler.sample("cpu"))
        else:
            self.hmc.initialize()

        if burn_in and self.n_burnin_steps:
            if log:
                logger.info(f"Burnin steps :{self.n_burnin_steps}")
            self.hmc.burnin(self.n_burnin_steps, self.n_traj_steps, self.step_size)

        return self.hmc.current_config

    def _step_python(self, config=None, record_observables=True):

        self.hmc.advance(
            1, n_traj_steps=self.n_traj_steps, step_size=self.step_size, config=config
        )
        self.current_config = self.hmc.current_config

        if record_observables:
            self.observables_rec.record_config(self.hmc.current_config[0])

    def _step_cpp(self, config=None, record_observables=True):

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

        if record_observables:
            self.observables_rec.record_obs(
                self.observable, self._trove[..., step_in_trove]
            )

    def autotune_step_size(self, desired_acceptance_percentage):

        n_autotune_samples = 1000
        tolerance = 0.05  # Tolerance
        step_size_min = 0.01 * self.initial_step_size
        step_size_max = 100 * self.initial_step_size
        converged = False
        tune_steps = 100

        pbar = tqdm(range(tune_steps))
        for _ in pbar:

            self.initialize(burn_in=False, log=False)
            self._step_size = 0.5 * (step_size_min + step_size_max)

            self.hmc.advance(
                n_steps=n_autotune_samples,
                n_traj_steps=self.n_traj_steps,
                step_size=self._step_size,
            )

            acceptance_rate = (
                self.hmc.acceptance_rate.mean()
                if isinstance(self.hmc.acceptance_rate, torch.Tensor)
                else self.hmc.acceptance_rate
            )
            if acceptance_rate > desired_acceptance_percentage:
                step_size_min = self._step_size
            else:
                step_size_max = self._step_size

            if abs(acceptance_rate - desired_acceptance_percentage) < tolerance:
                converged = True
                break

            pbar.set_description(
                f"step_size: {self._step_size}, Acceptance Rate {acceptance_rate}"
            )

        if not converged:
            self._step_size = self.initial_step_size


class HMC_PYTHON(object):
    def __init__(
        self,
        action,
        dim,
        batch_size: int = 1,
        bias: float = 0.0,
    ) -> None:

        self.action = action

        if not batch_size == 1:
            raise NotImplementedError("Batch size >1 currently not supported")

        self.batch_size = batch_size
        self.dim = dim

        self.bias = bias

    def reset_n_accepted(self):
        self.n_accepted = torch.zeros(self.batch_size, dtype=torch.float32)

    def initialize(self, initial_configs=Union[None, torch.Tensor]):

        self.reset_n_accepted()

        if initial_configs is not None:
            self.current_config = initial_configs.detach().clone()
        else:
            # bias breaks the Z_2 symmetry in initalization. This speeds up thermalization in broken phase.
            self.current_config = (
                torch.randn(self.batch_size, *self.dim).double() + self.bias
            )
            self.current_config = self.post_step(self.current_config)

        self.current_config = self.current_config.requires_grad_(False)

        self._configs = []
        self.n_steps_taken = 0

    @property
    def acceptance_rate(self):
        return self.n_accepted / self.n_steps_taken

    def advance(self, n_steps, n_traj_steps, step_size, config=None, log_configs=False):

        for _ in range(n_steps):
            self.step(n_traj_steps=n_traj_steps, step_size=step_size, config=config)

            if log_configs:
                self._configs += [self.current_config.clone().detach()]

        return self.current_config

    def reset_configs(self):
        self._configs = []

    def burnin(self, n_burnin_steps, n_traj_steps, step_size):

        for _ in range(n_burnin_steps):

            self.current_config = self.step(
                n_traj_steps=n_traj_steps, step_size=step_size
            )

        self.reset_n_accepted()

    def _leapfrog(self, q, p, step_size):
        p = p + step_size / 2 * self.action.force(q)
        q = q + p * step_size
        p = p + step_size / 2 * self.action.force(q)

        return q, p

    @property
    def configs(self):
        return self._configs

    @torch.no_grad()
    def step(self, n_traj_steps, step_size, config=None):

        if config is None:
            config = self.current_config

        q = config.clone()
        p = torch.randn(q.shape, device=config.device)

        h = 0.5 * torch.sum(p**2, dim=-1) + self.action.evaluate(q)

        for _ in range(n_traj_steps):
            q, p = self._leapfrog(q=q, p=p, step_size=step_size)

        hp = 0.5 * torch.sum(p**2, dim=-1) + self.action.evaluate(q)

        log_ratio = h - hp

        accept_mask = (log_ratio >= 0) | (
            torch.log(torch.rand_like(log_ratio)) < log_ratio
        )

        self.n_accepted += accept_mask.to(torch.device("cpu"))
        config[accept_mask] = q.detach()[accept_mask]
        config = config.detach()

        config = self.post_step(config)

        return config

    def post_step(self, config):
        return self.action.map_to_range(config)


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
