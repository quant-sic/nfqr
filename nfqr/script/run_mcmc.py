import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig, MCMCResult
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils import create_logger, setup_env

logger = create_logger(__name__)

def run_mcmc(args):
    
    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    mcmc_config = MCMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    result_config = MCMCResult(
        mcmc_config=mcmc_config,
        results=[],
    )

    if mcmc_config.out_dir.exists():
        logger.info("Result exists already. Skipping!")
        return

    mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
        **dict(mcmc_config)
    )
    mcmc.run()

    if args.get_stats:
        stats = mcmc.get_stats()
        stats["acc_rate"] = stats["acc_rate"].item()

        sus_exact = SusceptibilityExact(mcmc.action.beta, *mcmc_config.dim).evaluate()

        result_config.results.append({"stats": [stats], "sus_exact": sus_exact})

        result_config.save(mcmc_config.out_dir)



if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--redo_analysis", default=False, type=bool)
    parser.add_argument("--get_stats", default=False, type=bool)
    parser.add_argument("--skip_existing", default=True, type=bool)

    args = parser.parse_args()

    run_mcmc(args)

