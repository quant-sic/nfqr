import itertools
import shutil

from hmc import HMC
from tqdm.autonotebook import tqdm

from nfqr.globals import TEMP_DIR
from nfqr.mcmc.base import get_mcmc_statistics
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems.observable import ObservableRecorder
from nfqr.target_systems.rotor.rotor import QuantumRotor, TopologicalSusceptibility
from nfqr.target_systems.rotor.utils import SusceptibilityExact


def test_hmc_single_config():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [5, 10, 20]

    for beta, dim in tqdm(itertools.product(betas, dims)):

        target = TargetDensity.boltzmann_from_action(QuantumRotor(beta))

        tmp_test_path = TEMP_DIR / "test_hmc"
        if tmp_test_path.is_dir():
            shutil.rmtree(tmp_test_path)

        rec = ObservableRecorder(
            {"Chi_t": TopologicalSusceptibility()},
            save_dir_path=tmp_test_path,
            stats_function=get_mcmc_statistics,
        )

        hmc = HMC(
            observables_rec=rec,
            n_burnin_steps=100000,
            n_steps=250000,
            dim=dim,
            batch_size=1,
            target=target,
            autotune_step=True,
            n_traj_steps=20,
            alg="cpp_single",
            n_samples_at_a_time=25000,
        )

        hmc.run()
        stats = rec.aggregate()

        exact_sus = SusceptibilityExact(beta, dim).evaluate()

        assert abs(exact_sus - stats["Chi_t"]["mean"]) <= 3 * stats["Chi_t"]["error"]


def test_hmc_batch():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [5, 10, 20]

    for beta, dim in tqdm(itertools.product(betas, dims)):

        target = TargetDensity.boltzmann_from_action(QuantumRotor(beta))

        tmp_test_path = TEMP_DIR / "test_hmc"
        if tmp_test_path.is_dir():
            shutil.rmtree(tmp_test_path)

        rec = ObservableRecorder(
            {"Chi_t": TopologicalSusceptibility()},
            save_dir_path=tmp_test_path,
            stats_function=get_mcmc_statistics,
        )

        hmc = HMC(
            observables_rec=rec,
            n_burnin_steps=100000,
            n_steps=250000,
            dim=dim,
            batch_size=1,
            target=target,
            autotune_step=True,
            n_traj_steps=20,
            alg="cpp_batch",
            n_samples_at_a_time=25000,
        )

        hmc.run()
        stats = rec.aggregate()

        exact_sus = SusceptibilityExact(beta, dim).evaluate()

        assert abs(exact_sus - stats["Chi_t"]["mean"]) <= 3 * stats["Chi_t"]["error"]


#%%

from nfqr.mcmc.hmc.test_hmc import test_hmc_batch, test_hmc_single_config

test_hmc_single_config()
test_hmc_batch()
# %%
