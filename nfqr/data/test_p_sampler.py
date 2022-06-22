from pathlib import Path

import torch
from tqdm.auto import tqdm

from nfqr.data.config import ConditionConfig, MCMCSamplerConfig
from nfqr.data.datasampler import MCMCPSampler
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.target_systems import ActionConfig
from nfqr.target_systems.rotor import (
    RotorTrajectorySamplerConfig,
    SusceptibilityExact,
    TopologicalCharge,
    TopologicalSusceptibility,
)


def test_condition():
    batch_size = 5000
    dim = [10]

    p_value = 0.7

    for mcmc_alg in MCMC_REGISTRY._registry.keys():
        for mcmc_type in MCMC_REGISTRY[mcmc_alg]._registry.keys():

            mcmc_config = MCMCConfig(
                mcmc_alg=mcmc_alg,
                mcmc_type=mcmc_type,
                observables="Chi_t",
                n_steps=1,
                dim=dim,
                action_config=ActionConfig(beta=1.0),
                n_burnin_steps=500,
                n_traj_steps=3,
                hmc_engine="python",
                out_dir=Path("./"),
                initial_config_sampler_config=InitialConfigSamplerConfig(
                    trajectory_sampler_config=RotorTrajectorySamplerConfig(
                        dim=dim, traj_type="classical", k=1
                    )
                ),
            )

            mcmc_sampler_config_1 = MCMCSamplerConfig(
                mcmc_config=mcmc_config,
                condition_config=ConditionConfig(
                    params={
                        "type": "observable",
                        "target_system": "qr",
                        "value": [1.0],
                        "observable": "Q",
                    }
                ),
                batch_size=batch_size,
            )
            mcmc_sampler_config_0 = MCMCSamplerConfig(
                mcmc_config=mcmc_config,
                condition_config=ConditionConfig(
                    params={
                        "type": "observable",
                        "target_system": "qr",
                        "value": [0.0],
                        "observable": "Q",
                    }
                ),
                batch_size=batch_size,
            )

            p_sampler = MCMCPSampler(
                sampler_configs=[mcmc_sampler_config_0, mcmc_sampler_config_1],
                batch_size=10000,
                elements_per_dataset=25000,
                subset_distribution=[1 - p_value, p_value],
                num_workers=2,
            )

            samples = []
            num_batches = 10
            for res in range(num_batches):
                samples += [p_sampler.sample("cpu")]

            res = torch.round(TopologicalCharge().evaluate(torch.cat(samples))).mean()

            assert abs(res - p_value) <= (2 * num_batches / batch_size)


def test_sus_exact():
    batch_size = 5000
    dim = [10]
    beta = 1.5

    mcmc_sampler_config = MCMCSamplerConfig(
        mcmc_config=MCMCConfig(
            mcmc_alg="cluster",
            mcmc_type="wolff",
            observables="Chi_t",
            n_steps=1,
            dim=dim,
            action_config=ActionConfig(beta=beta),
            n_burnin_steps=1,
            n_traj_steps=3,
            out_dir=Path("./"),
        ),
        condition_config=ConditionConfig(),
        batch_size=batch_size,
    )

    p_sampler = MCMCPSampler(
        sampler_configs=[mcmc_sampler_config],
        batch_size=10000,
        elements_per_dataset=50000,
        subset_distribution=[1.0],
        num_workers=1,
        shuffle=False,
    )

    samples = []
    num_batches = 10
    for _ in tqdm(range(num_batches)):
        samples += [p_sampler.sample("cpu")]

    assert (
        abs(
            TopologicalSusceptibility().evaluate(torch.cat(samples)).mean()
            - SusceptibilityExact(beta=beta, D=10).evaluate()
        )
        < 0.1
    )


#%%
from nfqr.data.test_p_sampler import test_condition, test_sus_exact

test_condition()
test_sus_exact()
