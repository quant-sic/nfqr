import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc.config import MCMCConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY, HMCConfig
from nfqr.target_systems.rotor.utils import SusceptibilityExact

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    hmc_config = HMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    hmc = HMC_REGISTRY[hmc_config.hmc_type](**dict(hmc_config))
    hmc.run()

    stats = hmc.get_stats()

    sus_exact = SusceptibilityExact(hmc_config.action_config.beta,hmc_config.dim)

    result_config = MCMCConfig(
        hmc_type=hmc_config.hmc_type,
        observables=hmc_config.observables,
        acceptance_rate=stats["acceptance_rate"],
        n_steps=stats["n_steps"],
        obs_stats=stats["obs_stats"],
        sus_exact=sus_exact
    )

    result_config.save(hmc_config.out_dir)
