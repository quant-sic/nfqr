import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC, MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig, MCMCResult
from nfqr.recorder import ObservableRecorder
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def error_reached_stop_condition(stats_list, threshold):

    errors = defaultdict(list)

    for obs_stats in [s["obs_stats"] for s in stats_list]:
        for obs, stats in obs_stats.items():
            errors[obs].append(stats["error"])

    means = {obs: (sum(e) / len(e)) for obs, e in errors.items()}
    return max(means.values()) < threshold


def default_stop_condition(stats_list, *args, **kwargs):
    return True


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--redo_analysis", default=False, type=bool)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    mcmc_config = MCMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    if not (mcmc_config.out_dir / "mcmc_result.json").is_file():
        result_config = MCMCResult(
            mcmc_config=mcmc_config,
            results=[],
        )
        n_steps_done = []
        obs_done_dict = {}

    else:
        result_config = MCMCResult.from_directory(mcmc_config.out_dir)
        n_steps_done, obs_done_list = map(
            list,
            zip(
                *[
                    (
                        int(r["stats"][0]["n_steps"]),
                        list(r["stats"][0]["obs_stats"].keys()),
                    )
                    for r in result_config.results
                ]
            ),
        )
        obs_done_dict = {n: obs_l for n, obs_l in zip(n_steps_done, obs_done_list)}

    if mcmc_config.min_error is not None:
        iterator = zip(
            map(int, mcmc_config.n_steps),
            cycle(
                (
                    partial(
                        error_reached_stop_condition, threshold=mcmc_config.min_error
                    ),
                )
            ),
        )
    else:
        assert isinstance(mcmc_config.n_steps, int)
        iterator = ((mcmc_config.n_steps, default_stop_condition),)

    logger.info(n_steps_done)

    for n_steps, condition in iterator:

        if n_steps in n_steps_done:
            logger.info(f"N steps {n_steps} already done. Skipping!")
            continue

        logger.info(f"Starting Run for N steps {n_steps}")

        run_config = mcmc_config.copy()

        n_steps_done.append(n_steps)
        run_config.n_steps = n_steps
        stats_list = []

        for repeat_idx in range(mcmc_config.n_repeat):

            run_config.out_dir = mcmc_config.out_dir / "{:d}/{:d}".format(
                n_steps, repeat_idx
            )

            mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
                **dict(run_config)
            )
            mcmc.run()
            stats = mcmc.get_stats()
            stats["acc_rate"] = stats["acc_rate"].item()
            stats_list.append(stats)

        sus_exact = SusceptibilityExact(mcmc.action.beta, *run_config.dim).evaluate()

        logger.info(stats_list)

        result_config.results.append({"stats": stats_list, "sus_exact": sus_exact})

        result_config.save(mcmc_config.out_dir)

        if condition(stats_list=stats_list):
            break
