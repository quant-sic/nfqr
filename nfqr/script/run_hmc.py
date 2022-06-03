import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc.config import MCMCConfig
from nfqr.mcmc.hmc.hmc import HMC_REGISTRY, HMCConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    hmc_config = HMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )


    if Path(hmc_config.out_dir).is_dir() and (Path(hmc_config.out_dir)/"mcmc_result.json").is_file():
        logger.info("Experiment already run successfully. Aborting")
    else:
        hmc = HMC_REGISTRY[hmc_config.hmc_type](**dict(hmc_config))
        hmc.run()

        stats = hmc.get_stats()

        sus_exact = SusceptibilityExact(
            hmc_config.action_config.beta, hmc_config.dim
        ).evaluate()

        result_config = MCMCConfig(
            hmc_type=hmc_config.hmc_type,
            observables=hmc_config.observables,
            acceptance_rate=stats["acc_rate"],
            n_steps=stats["n_steps"],
            obs_stats=stats["obs_stats"],
            sus_exact=sus_exact,
        )

        result_config.save(hmc_config.out_dir)
