import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nfqr.eval.evaluation import EvalConfig, EvalResult, run_nmcmc_n_replicas_for_model
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.mcmc.nmcmc import NeuralMCMC, NeuralMCMCParallel
from nfqr.mcmc.stats import get_mcmc_statistics
from nfqr.recorder import ObservableRecorder
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils import create_logger, setup_env
from nfqr.utils.tensorboard import EventAccumulatorHook

logger = create_logger(__name__)


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


def eval_single_runs_result_config(task_dir, eval_config, lit_model):

    if not (task_dir / "eval_result.json").is_file():
        eval_result = EvalResult(
            observables=eval_config.observables,
            n_samples=[],
        )
        stats_nip_list = []
        stats_nmcmc_list = []
        n_samples = []

    else:
        eval_result = EvalResult.from_directory((task_dir / "eval_result.json").parent)

        assert set(eval_config.observables).issubset(
            set(eval_result.observables)
        ) and set(eval_result.observables).issubset(set(eval_config.observables))

        n_samples = eval_result.n_samples.copy()
        stats_nip_list = eval_result.nip.copy()
        stats_nmcmc_list = eval_result.nmcmc.copy()

    for n_iter, batch_size in zip(eval_config.n_iter, eval_config.batch_size):

        use_idx = None
        if n_iter * batch_size in n_samples:
            idx = n_samples.index(n_iter * batch_size)
            try:
                check_list = filter(
                    lambda _list: len(_list) > 0, (stats_nip_list, stats_nmcmc_list)
                ).__next__()
            except StopIteration:
                raise RuntimeError("No results found but steps set!")

            if not isinstance(check_list[idx], list) and eval_config.n_repeat > 1:
                use_idx = idx

            elif (
                isinstance(check_list[idx], list)
                and len(check_list[idx]) >= eval_config.n_repeat
            ):
                logger.info(
                    f"N samples {n_iter* batch_size} already exceuted: skipping!"
                )
                continue
            elif (
                isinstance(check_list[idx], list)
                and len(check_list[idx]) < eval_config.n_repeat
            ):
                use_idx = idx

        if use_idx is None:
            n_samples.append(n_iter * batch_size)

        logger.info(f"Model Sus exakt {lit_model.sus_exact}")
        logger.info(f"Model beta {lit_model.target.dist.action.beta}")

        logger.info(f"Executing for N samples {n_iter* batch_size}!")

        nip_repeat = []
        nmcmc_repeat = []

        if "nmcmc" in eval_config.methods:
            for repeat_idx in range(eval_config.n_repeat):

                stats_nmcmc = lit_model.estimate_obs_nmcmc(
                    batch_size=batch_size, n_iter=n_iter
                )
                nmcmc_repeat.append(stats_nmcmc)

            if use_idx is not None:
                stats_nmcmc_list[use_idx] = nmcmc_repeat
            else:
                stats_nmcmc_list.append(nmcmc_repeat)

        if "nip" in eval_config.methods:
            for repeat_idx in range(eval_config.n_repeat):

                stats_nip = lit_model.estimate_obs_nip(
                    batch_size=batch_size, n_iter=n_iter, ess_p=False
                )
                nip_repeat.append(stats_nip)

            if use_idx is not None:
                stats_nip_list[use_idx] = nip_repeat
            else:
                stats_nip_list.append(nip_repeat)

        if n_iter * batch_size > 100000:
            logger.info(stats_nmcmc_list)
            try:
                relative_error = (
                    np.array(
                        [stats["obs_stats"]["Chi_t"]["error"] for stats in nmcmc_repeat]
                    )
                    / np.array(
                        [stats["obs_stats"]["Chi_t"]["mean"] for stats in nmcmc_repeat]
                    )
                ).mean()

                if relative_error < eval_config.max_rel_error:
                    break
            except ZeroDivisionError:
                pass

        eval_result.exact_sus = lit_model.sus_exact
        eval_result.nip = stats_nip_list
        eval_result.nmcmc = stats_nmcmc_list
        eval_result.n_samples = n_samples

        eval_result.save(task_dir)


def nmcmc_tau_int(lit_model, run_dir, chain_length=10**5, n_replicas=5):

    run_nmcmc_n_replicas_for_model(
        model=lit_model.model,
        observables=lit_model.observables,
        target=lit_model.target,
        trove_size=10000,
        out_dir=run_dir / "tau_int_run",
        n_steps=chain_length,
        n_replicas=n_replicas,
    )

    rec = ObservableRecorder(
        observables=lit_model.observables_fn,
        save_dir_path=run_dir / "tau_int_run",
        delete_existing_data=False,
        n_replicas=n_replicas,
    )
    data = rec["Chi_t"]

    stats = get_mcmc_statistics(data)

    return stats["tau_int"]


def eval_increment_nmcmc(task_dir, lit_model, eval_config):

    tau_int = nmcmc_tau_int(lit_model, run_dir=task_dir)
    tau_int = int(tau_int) if tau_int > 1 else 1

    EvalResult(
        observables=eval_config.observables, n_samples=[], skipped_steps=tau_int
    ).save(task_dir)

    results_df = pd.DataFrame(
        columns=["stats"],
        index=pd.MultiIndex.from_product(
            (
                range(eval_config.n_replicas),
                range(
                    eval_config.min_tau_int_stats_length,
                    int(eval_config.max_stats_eval / tau_int),
                    eval_config.stats_tau_int_step_interval,
                ),
            ),
        ),
    )

    if tau_int is None:
        raise ValueError("tau_int not found.")
    else:
        logger.info(f"Using tau_int = {tau_int}")

    lit_model.model.eval()
    lit_model.model.double()

    nmcmc = NeuralMCMCParallel(
        model=lit_model.model,
        target=lit_model.target,
        trove_size=10000,
        n_steps=eval_config.min_tau_int_stats_length * tau_int,
        observables=lit_model.observables,
        out_dir=task_dir / "nmcmc",
        n_replicas=eval_config.n_replicas,
        n_record_skips=tau_int,
    )
    nmcmc.run()

    nmcmc.stats_skip_steps = eval_config.stats_tau_int_skip_steps

    logger.info(f"Starting {eval_config.stats_method} error analysis.")

    converged_chains = []

    steps_bar = tqdm(results_df.index.levels[1])
    for steps_idx, n_steps_stats in enumerate(steps_bar):

        nmcmc.continue_for_nsteps(
            eval_config.stats_tau_int_step_interval * tau_int, disable_tqdm=True
        )

        for eval_idx in list(set(results_df.index.levels[0]) - set(converged_chains)):

            nmcmc.eval_idx = eval_idx
            nmcmc.stats_limit = n_steps_stats

            if eval_config.stats_method == "wolff":
                stats = nmcmc.get_stats()

            if isinstance(stats["acc_rate"], torch.Tensor):
                stats["acc_rate"] = stats["acc_rate"].item()

            try:
                relative_error = (
                    stats["obs_stats"]["Chi_t"]["error"]
                    / stats["obs_stats"]["Chi_t"]["mean"]
                )

                steps_bar.set_description(
                    "Relative error: {:.2e} at steps {:d} . Target {:.2}. Tau_int: {:.2f}. Converged Chains {}".format(
                        relative_error,
                        n_steps_stats,
                        eval_config.max_rel_error,
                        stats["obs_stats"]["Chi_t"]["tau_int"],
                        converged_chains,
                    )
                )

                if n_steps_stats > 1000:
                    if relative_error < eval_config.max_rel_error:
                        converged_chains.append(eval_idx)
            except ZeroDivisionError:
                pass

            results_df.loc[(eval_idx, n_steps_stats), "stats"] = (
                stats["obs_stats"]["Chi_t"]["mean"],
                stats["obs_stats"]["Chi_t"]["error"],
            )

        results_df.to_pickle(task_dir / f"results_df_{eval_config.stats_method}.pkl")

        if len(converged_chains) == eval_config.n_replicas:
            break


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    logger.info(f"Starting Evaluation task for exp dir {exp_dir}")

    setup_env()

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
    )
    eval_config = EvalConfig.from_directory(exp_dir)

    log_dir = "task_{}".format(os.environ["task_id"])

    ckpts, steps = zip(
        *map(
            lambda path: (
                path,
                int(re.search("step=([0-9]*).", path.name).groups()[0]),
            ),
            filter(
                lambda p: "step" in p.name,
                (exp_dir / f"logs/{log_dir}").glob("**/*.ckpt"),
            ),
        )
    )

    pbar = tqdm(map(lambda idx: ckpts[idx], np.argsort(steps)))
    for model_ckpt_path in pbar:

        if (eval_config.models is not None) and (
            model_ckpt_path.stem not in eval_config.models
        ):
            continue

        elif model_ckpt_path.stem == "model":
            continue

        pbar.set_description(
            "Evaluation for task {} and model {}".format(
                os.environ["task_id"], model_ckpt_path.stem
            )
        )

        task_dir = exp_dir / f"eval/{log_dir}/{model_ckpt_path.stem}"

        step = int(re.search("step=([0-9]*).", model_ckpt_path.name).groups()[0])

        if step > eval_config.max_step or step < eval_config.min_step:
            continue

        if "beta_scheduled" in str(model_ckpt_path):

            events_file_path = (
                (model_ckpt_path.parent.parent.parent)
                .glob("events*[!.hook]")
                .__next__()
            )

            acc = EventAccumulatorHook(events_file_path)

            _, step_nums, vals = acc.Scalars("beta")

            plot_step = np.argwhere((step_nums >= (step - 1)).cumsum(axis=0) == 1)[0][0]

            beta = vals[plot_step]

            if eval_config.start_from_target_beta:
                if (
                    not beta
                    == train_config.trainer_configs[0]
                    .scheduler_configs[0]
                    .specific_scheduler_config.target_beta
                ):
                    logger.info(
                        f"beta {beta} does not match target beta {train_config.trainer_configs[0].scheduler_configs[0].specific_scheduler_config.target_beta}"
                    )
                    continue

            train_config.action_config.specific_action_config.beta = beta
            logger.info(f"Setting beta to {beta}")

        model_kwargs_dict = dict(train_config)
        model_kwargs_dict.update({"trainer_config": train_config.trainer_configs[-1]})
        lit_model = LitFlow.load_from_checkpoint(
            model_ckpt_path, **model_kwargs_dict, mode="eval"
        )

        if eval_config.mode == "discrete_runs":
            eval_single_runs_result_config(
                task_dir=task_dir, lit_model=lit_model, eval_config=eval_config
            )
        elif eval_config.mode == "increment_nmcmc":
            eval_increment_nmcmc(
                task_dir=task_dir, lit_model=lit_model, eval_config=eval_config
            )
