import itertools

from hmc import HMC
from tqdm import tqdm

from nfqr.globals import TMP_DIR
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.target_systems import ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.target_systems.rotor.action import QuantumRotorConfig
from nfqr.target_systems.rotor.trajectories_samplers import RotorTrajectorySamplerConfig


def test_hmc_single_config():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [[5], [10], [20]]

    for beta, dim in tqdm(itertools.product(betas, dims)):

        hmc = HMC(
            observables=["Chi_t"],
            n_burnin_steps=100000,
            n_steps=250000,
            dim=dim,
            autotune_step=True,
            n_traj_steps=20,
            hmc_engine="cpp_single",
            n_samples_at_a_time=25000,
            out_dir=TMP_DIR / "test_hmc",
            action_config=ActionConfig(
                specific_action_config=QuantumRotorConfig(beta=beta),
                action_type="qr",
                target_system="qr",
            ),
            initial_config_sampler_config=InitialConfigSamplerConfig(
                trajectory_sampler_config=RotorTrajectorySamplerConfig(
                    traj_type="hot", dim=dim
                )
            ),
        )

        hmc.run()
        stats = hmc.get_stats()

        exact_sus = SusceptibilityExact(beta, dim[0]).evaluate()

        assert (
            abs(exact_sus - stats["obs_stats"]["Chi_t"]["mean"])
            <= 3 * stats["obs_stats"]["Chi_t"]["error"]
        )


def test_hmc_python():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [[5], [10], [20]]
    n_replicas_list = [1, 5]

    p_bar = tqdm(itertools.product(betas, dims, n_replicas_list))
    for beta, dim, n_replicas in p_bar:
        p_bar.set_description(f"n_replicas: {n_replicas}, dim: {dim}, beta: {beta}")

        hmc = HMC(
            observables=["Chi_t"],
            n_burnin_steps=10000,
            n_steps=25000,
            dim=dim,
            autotune_step=True,
            n_traj_steps=20,
            hmc_engine="python",
            n_samples_at_a_time=50000,
            out_dir=TMP_DIR / "test_hmc",
            action_config=ActionConfig(
                specific_action_config=QuantumRotorConfig(beta=beta),
                action_type="qr",
                target_system="qr",
            ),
            initial_config_sampler_config=InitialConfigSamplerConfig(
                trajectory_sampler_config=RotorTrajectorySamplerConfig(
                    traj_type="hot", dim=dim
                )
            ),
            n_replicas=n_replicas,
        )

        hmc.run()
        stats = hmc.get_stats()
        exact_sus = SusceptibilityExact(beta, dim[0]).evaluate()

        print(stats, exact_sus)

        assert (
            abs(exact_sus - stats["obs_stats"]["Chi_t"]["mean"])
            <= 3 * stats["obs_stats"]["Chi_t"]["error"]
        )


def test_hmc_batch():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [[5], [10], [20]]
    n_replicas_list = [1, 5]

    p_bar = tqdm(itertools.product(betas, dims, n_replicas_list))
    for beta, dim, n_replicas in p_bar:
        p_bar.set_description(f"n_replicas: {n_replicas}, dim: {dim}, beta: {beta}")

        hmc = HMC(
            observables=["Chi_t"],
            n_burnin_steps=10000,
            n_steps=50000,
            dim=dim,
            autotune_step=True,
            n_traj_steps=20,
            hmc_engine="cpp_batch",
            n_samples_at_a_time=25000,
            out_dir=TMP_DIR / "test_hmc",
            action_config=ActionConfig(
                specific_action_config=QuantumRotorConfig(beta=beta),
                action_type="qr",
                target_system="qr",
            ),
            initial_config_sampler_config=InitialConfigSamplerConfig(
                trajectory_sampler_config=RotorTrajectorySamplerConfig(
                    traj_type="hot", dim=dim
                )
            ),
            n_replicas=n_replicas,
        )

        hmc.run()
        stats = hmc.get_stats()
        exact_sus = SusceptibilityExact(beta, dim[0]).evaluate()

        print(stats, exact_sus)

        assert (
            abs(exact_sus - stats["obs_stats"]["Chi_t"]["mean"])
            <= 3 * stats["obs_stats"]["Chi_t"]["error"]
        )


#%%

from nfqr.mcmc.hmc.test_hmc import test_hmc_batch, test_hmc_single_config

test_hmc_batch()
test_hmc_python()
test_hmc_single_config()
# %%
