import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc.config import MCMCConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY, HMCConfig

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    hmc_config = HMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    hmc = HMC_REGISTRY[hmc_config.hmc_type](**hmc_config)
    hmc.run()

    stats = hmc.get_stats()

    result_config = MCMCConfig(
        hmc_type=hmc_config.hmc_type,
        observables=hmc_config.observables,
        acceptance_rate=stats["acceptance_rate"],
        n_steps=stats["n_steps"],
        obs_stats=stats["obs_stats"],
    )

    result_config.save(hmc_config.out_dir)
