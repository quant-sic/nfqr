import os
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from nfqr.eval.evaluation import EvalConfig, EvalResult
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils import create_logger
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re

logger = create_logger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
        num_tasks=int(os.environ["num_tasks"]),
    )
    eval_config = EvalConfig.from_directory(exp_dir)

    log_dir = "task_{}".format(os.environ["task_id"])

    pbar = tqdm((exp_dir / f"logs/{log_dir}").glob("**/*.ckpt"))
    for model_ckpt_path in pbar:

        if eval_config.models is not None and not model_ckpt_path.stem in eval_config.models:
            continue

        pbar.set_description(
            "Evaluation for task {} and model {}".format(
                os.environ["task_id"], model_ckpt_path.stem
            )
        )

        task_dir = exp_dir / f"eval/{log_dir}/{model_ckpt_path.stem}"

        if "beta_scheduled" in str(model_ckpt_path):
            events_file_path = (model_ckpt_path.parent.parent.parent).glob("events*").__next__()

            acc = EventAccumulator(str(events_file_path)).Reload()

            _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]

            step = int(re.search("step=([0-9]*).",model_ckpt_path.name).groups()[0])
            
            plot_step = np.argwhere((step_nums>=(step-1)).cumsum(axis=0)==1)[0][0]

            beta = vals[plot_step]

            train_config.action_config.specific_action_config.beta = beta
            logger.info(f"Setting beta to {beta}")


        model_kwargs_dict = dict(train_config)
        model_kwargs_dict.update({"trainer_config":train_config.trainer_configs[-1]})
        lit_model = LitFlow.load_from_checkpoint(
            model_ckpt_path, **model_kwargs_dict, mode="eval"
        )

        if not (task_dir/"eval_result.json").is_file():
            eval_result = EvalResult(
                observables=eval_config.observables,
                n_samples=[],
            )
            stats_nip_list = []
            stats_nmcmc_list = []
            n_samples = []

        else:
            eval_result = EvalResult.from_directory((task_dir/"eval_result.json").parent)

            assert set(eval_config.observables).issubset(set(eval_result.observables)) and set(eval_result.observables).issubset(set(eval_config.observables))

            n_samples = eval_result.n_samples.copy()
            stats_nip_list = eval_result.nip.copy()
            stats_nmcmc_list = eval_result.nmcmc.copy()


        for n_iter, batch_size in zip(eval_config.n_iter, eval_config.batch_size):

            if n_iter * batch_size in n_samples:
                continue
            else:
                n_samples.append(n_iter * batch_size)


            if "nip" in eval_config.methods:
                stats_nip = lit_model.estimate_obs_nip(
                    batch_size=batch_size, n_iter=n_iter
                )
                stats_nip_list += [stats_nip]

            if "nmcmc" in eval_config.methods:
                stats_nmcmc = lit_model.estimate_obs_nmcmc(
                    batch_size=batch_size, n_iter=n_iter
                )
                stats_nmcmc_list += [stats_nmcmc]


            eval_result.exact_sus = lit_model.sus_exact
            eval_result.nip = stats_nip_list
            eval_result.nmcmc = stats_nmcmc_list
            eval_result.n_samples = n_samples

            eval_result.save(task_dir)
