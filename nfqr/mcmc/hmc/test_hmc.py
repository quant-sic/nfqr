import itertools
import shutil

from hmc import HMC
from tqdm.autonotebook import tqdm

from nfqr.globals import TMP_DIR
from nfqr.target_systems import ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact


def test_hmc_single_config():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [5, 10, 20]

    for beta, dim in tqdm(itertools.product(betas, dims)):

        tmp_test_path = TMP_DIR / "test_hmc"
        if tmp_test_path.is_dir():
            shutil.rmtree(tmp_test_path)

        hmc = HMC(
            observables=["Chi_t"],
            n_burnin_steps=100000,
            n_steps=250000,
            dim=dim,
            batch_size=1,
            autotune_step=True,
            n_traj_steps=20,
            alg="cpp_single",
            n_samples_at_a_time=25000,
            out_dir=tmp_test_path,
            action="qr",
            action_config=ActionConfig(beta=beta),
        )

        hmc.run()
        stats = hmc.observables_rec.aggregate()

        exact_sus = SusceptibilityExact(beta, dim).evaluate()

        assert abs(exact_sus - stats["Chi_t"]["mean"]) <= 3 * stats["Chi_t"]["error"]


def test_hmc_batch():

    betas = [0.1, 0.2, 0.3, 0.5]
    dims = [5, 10, 20]

    for beta, dim in tqdm(itertools.product(betas, dims)):

        tmp_test_path = TMP_DIR / "test_hmc"
        if tmp_test_path.is_dir():
            shutil.rmtree(tmp_test_path)

        hmc = HMC(
            observables=["Chi_t"],
            n_burnin_steps=100000,
            n_steps=250000,
            dim=dim,
            batch_size=1,
            autotune_step=True,
            n_traj_steps=20,
            alg="cpp_batch",
            n_samples_at_a_time=25000,
            out_dir=tmp_test_path,
            action="qr",
            action_config=ActionConfig(beta=beta),
        )

        hmc.run()
        stats = hmc.observables_rec.aggregate()

        exact_sus = SusceptibilityExact(beta, dim).evaluate()

        assert abs(exact_sus - stats["Chi_t"]["mean"]) <= 3 * stats["Chi_t"]["error"]


#%%

from nfqr.mcmc.hmc.test_hmc import test_hmc_batch, test_hmc_single_config

test_hmc_single_config()
test_hmc_batch()
# %%
