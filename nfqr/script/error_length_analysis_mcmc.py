import os
from argparse import ArgumentParser
from functools import partialmethod
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig, MCMCResult
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.utils import create_logger, setup_env

logger = create_logger(__name__)


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
            (range(mcmc_config.n_replicas), range(100, mcmc_config.max_stats_eval, mcmc_config.stats_step_interval)),
        ),
    )

    for eval_idx in results_df.index.levels[0]:
        mcmc.eval_idx = eval_idx

        for n_steps_stats in tqdm(results_df.index.levels[1]):

            mcmc.stats_limit = n_steps_stats

            stats = mcmc.get_stats()
            if isinstance(stats["acc_rate"], torch.Tensor):
                stats["acc_rate"] = stats["acc_rate"].item()

            if n_steps_stats>100000:
                try:
                    relative_error = stats["obs_stats"]["Chi_t"]["error"]/stats["obs_stats"]["Chi_t"]["mean"]
                    if relative_error<mcmc_config.min_error:
                        break
                except ZeroDivisionError:
                    pass

            results_df.loc[(eval_idx, n_steps_stats), "stats"] = (stats["obs_stats"][
                "Chi_t"
            ]["mean"],stats["obs_stats"][
                "Chi_t"
            ]["error"])

        results_df.to_pickle(mcmc_config.out_dir / "results_df.pkl")
