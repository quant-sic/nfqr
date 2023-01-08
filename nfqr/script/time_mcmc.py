import os
from argparse import ArgumentParser
from pathlib import Path

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils.misc import create_logger
from nfqr.utils import setup_env

import timeit

logger = create_logger(__name__)

def run_mcmc(mcmc,n_steps):

    for _ in range(n_steps):
        mcmc.step()
        mcmc.n_current_steps += 1

if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--redo_analysis",default=False, type=bool)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    mcmc_config = MCMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )


    mcmc_config.n_burnin_steps=0

    n_steps=100000

    mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
        **dict(mcmc_config)
    )

    mcmc.initialize()
    mcmc.step_size

    # time this

    timer = timeit.Timer(stmt='run_mcmc(mcmc,n_steps)', setup='pass',globals={**globals(),**locals()})
    time_res = timer.repeat(repeat=3,number=3)

    print(time_res,n_steps)

    time_res = timer.repeat(repeat=3,number=1)

    print(time_res,n_steps)

