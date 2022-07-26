import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig, MCMCResult
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils.misc import create_logger
from nfqr.recorder import ObservableRecorder
from nfqr.mcmc import MCMC

logger = create_logger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--redo_analysis",default=False, type=bool)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    mcmc_config = MCMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    if (
        Path(mcmc_config.out_dir).is_dir()
        and (Path(mcmc_config.out_dir) / "mcmc_result.json").is_file() and not args.redo_analysis
    ):
        logger.info("Experiment already run successfully. Aborting")
    else:

        if args.redo_analysis:
            logger.info("Redoing analysis")
            mcmc = MCMC(observables=mcmc_config.observables,n_steps=mcmc_config.n_steps,target_system=mcmc_config.action_config.target_system,out_dir=mcmc_config.out_dir,n_replicas=mcmc_config.n_replicas,delete_existing_data=False)
            stats = mcmc.get_stats()
        else:
            mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
                **dict(mcmc_config)
            )
            mcmc.run()
            stats = mcmc.get_stats()

        sus_exact = SusceptibilityExact(
            mcmc_config.action_config.specific_action_config.beta, *mcmc_config.dim
        ).evaluate()

        result_config = MCMCResult(
            mcmc_config=mcmc_config,
            acceptance_rate=stats["acc_rate"],
            n_steps=stats["n_steps"],
            obs_stats=stats["obs_stats"],
            sus_exact=sus_exact,
        )

        result_config.save(mcmc_config.out_dir)
