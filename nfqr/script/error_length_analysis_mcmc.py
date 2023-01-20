import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig, MCMCResult
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils import create_logger, setup_env

logger = create_logger(__name__)


def exp_fit_function(x, a, b):
    return a * x**b


def get_tau_int(beta, dim):

    autocorrelations_dir = EXPERIMENTS_DIR / "mcmc/hmc/clp_2/mcmc"

    for task_dir_path in autocorrelations_dir.glob("*"):
        try:
            result = MCMCResult.from_directory(task_dir_path)
        except FileNotFoundError:
            continue

        target = TargetDensity.boltzmann_from_action(
            ACTION_REGISTRY[result.mcmc_config.action_config.target_system][
                result.mcmc_config.action_config.action_type
            ](**dict(result.mcmc_config.action_config.specific_action_config))
        )

        if target.dist.action.beta == beta and result.mcmc_config.dim[0] == dim:
            return int(result.obs_stats["Chi_t"]["tau_int"])

    return int(exp_fit_function(beta ** (-1), 61.51495138, -6.11572102))


def block_statistics(data, tau_int, factor: float = 2, idx_in_block=0):

    block_size = tau_int * factor
    splits = torch.split(data, split_size_or_sections=block_size)
    data_subsampled_list = []
    for split in splits:
        data_subsampled_list.append(split[idx_in_block])

    mean_block = torch.tensor(data_subsampled_list).mean()
    std_block = torch.tensor(data_subsampled_list).std() / np.sqrt(
        len(data_subsampled_list)
    )

    stats = defaultdict(lambda: defaultdict(dict))
    stats["obs_stats"]["Chi_t"]["mean"] = mean_block
    stats["obs_stats"]["Chi_t"]["error"] = std_block

    return stats


if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    mcmc_config = MCMCConfig.from_directory_for_task(
        exp_dir, task_id=int(os.environ["task_id"])
    )

    result_config = MCMCResult(
        mcmc_config=mcmc_config,
        results=[],
    )

    mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
        **dict(mcmc_config), delete_existing_data=False
    )

    sus_exact = SusceptibilityExact(mcmc.action.beta, *mcmc_config.dim).evaluate()

    results_df = pd.DataFrame(
        columns=["stats"],
        index=pd.MultiIndex.from_product(
            (
                range(mcmc_config.n_replicas),
                range(
                    mcmc_config.min_stats_length,
                    mcmc_config.max_stats_eval,
                    mcmc_config.stats_step_interval,
                ),
            ),
        ),
    )
    tau_int = get_tau_int(mcmc.action.beta, mcmc_config.dim[0])
    if tau_int is None:
        raise ValueError("tau_int not found.")
    else:
        logger.info(f"Using tau_int = {tau_int}")

    logger.info(f"Starting {mcmc_config.stats_method} error analysis.")
    for eval_idx in results_df.index.levels[0]:
        mcmc.eval_idx = eval_idx

        data = mcmc.observables_rec.__getitem__("Chi_t", rep_idx=mcmc.eval_idx)

        steps_bar = tqdm(results_df.index.levels[1])
        for n_steps_stats in steps_bar:
            # logger.info(n_steps_stats)
            mcmc.stats_limit = n_steps_stats
            mcmc.stats_skip_steps = mcmc_config.stats_skip_steps

            if mcmc_config.stats_method == "wolff":
                stats = mcmc.get_stats()
            elif mcmc_config.stats_method == "blocked":

                stats = block_statistics(data=data[:n_steps_stats], tau_int=tau_int)

            if isinstance(stats["acc_rate"], torch.Tensor):
                stats["acc_rate"] = stats["acc_rate"].item()

            if n_steps_stats > 0.1 * mcmc_config.max_stats_eval:
                try:
                    relative_error = (
                        stats["obs_stats"]["Chi_t"]["error"]
                        / stats["obs_stats"]["Chi_t"]["mean"]
                    )

                    steps_bar.set_description(
                        "Relative error: {:.2e} at steps {} . Target {:.2}. Tau_int: {:.2f}.".format(
                            relative_error,
                            n_steps_stats,
                            mcmc_config.min_error,
                            stats["obs_stats"]["Chi_t"]["tau_int"],
                        )
                    )

                    if relative_error < mcmc_config.min_error:
                        break
                except ZeroDivisionError:
                    pass

            results_df.loc[(eval_idx, n_steps_stats), "stats"] = (
                stats["obs_stats"]["Chi_t"]["mean"],
                stats["obs_stats"]["Chi_t"]["error"],
            )

        results_df.to_pickle(
            mcmc_config.out_dir / f"results_df_{mcmc_config.stats_method}.pkl"
        )
