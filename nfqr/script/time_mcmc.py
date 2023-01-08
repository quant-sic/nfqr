import json
import os
import timeit
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils import setup_env
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def run_mcmc(mcmc, n_steps):

    for _ in range(n_steps):
        mcmc.step()
        mcmc.n_current_steps += 1


if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--redo_analysis", default=False, type=bool)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    for task_id in tqdm(range(20)):

        os.environ["task_id"] = str(task_id)
        mcmc_config = MCMCConfig.from_directory_for_task(
            exp_dir, task_id=int(os.environ["task_id"])
        )

        mcmc_config.n_burnin_steps = 0

        n_steps = 100000

        mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
            **dict(mcmc_config)
        )

        mcmc.initialize()
        mcmc.step_size

        # time this

        timer = timeit.Timer(
            stmt="run_mcmc(mcmc,n_steps)",
            setup="pass",
            globals={**globals(), **locals()},
        )
        time_res = timer.repeat(repeat=3, number=1)
        time_res_dict = {"result": time_res, "n_steps": n_steps}

        save_dir = exp_dir / f"timed/task_{task_id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "result.json", "w") as file:
            json.dump(time_res_dict, file)
