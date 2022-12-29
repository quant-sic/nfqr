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
from nfqr.utils import setup_env

logger = create_logger(__name__)

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
        num_tasks=int(os.environ["num_tasks"]),
    )
    eval_config = EvalConfig.from_directory(exp_dir)

    log_dir = "task_{}".format(os.environ["task_id"])

    pbar = tqdm((exp_dir / f"logs/{log_dir}").glob("**/*.ckpt"))
    for model_ckpt_path in pbar:

        if (eval_config.models is not None) and (not model_ckpt_path.stem in eval_config.models):
            continue
        elif model_ckpt_path.stem == "model":
            continue

        pbar.set_description(
            "Evaluation for task {} and model {}".format(
                os.environ["task_id"], model_ckpt_path.stem
            )
        )

        task_dir = exp_dir / f"eval/{log_dir}/{model_ckpt_path.stem}"

        step = int(re.search("step=([0-9]*).",model_ckpt_path.name).groups()[0])

        if step>eval_config.max_step or step<eval_config.min_step:
            continue

        if "beta_scheduled" in str(model_ckpt_path):
            events_file_path = (model_ckpt_path.parent.parent.parent).glob("events*").__next__()

            acc = EventAccumulator(str(events_file_path)).Reload()

            _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]
            
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

            use_idx = None
            if n_iter * batch_size in n_samples:
                idx = n_samples.index(n_iter * batch_size)
                try:
                    check_list = filter(lambda l:len(l)>0,(stats_nip_list,stats_nmcmc_list)).__next__()
                except StopIteration:
                    raise RuntimeError("No results found but steps set!")

                if not isinstance(check_list[idx],list) and eval_config.n_repeat>1:
                    use_idx = idx
                    
                elif isinstance(check_list[idx],list) and len(check_list[idx])>=eval_config.n_repeat:
                    logger.info(f"N samples {n_iter* batch_size} already exceuted: skipping!")
                    continue
                elif isinstance(check_list[idx],list) and len(check_list[idx])<eval_config.n_repeat:
                    use_idx = idx
            
            if use_idx is None:
                n_samples.append(n_iter * batch_size)

            logger.info(f"Model Sus exakt {lit_model.sus_exact}")
            logger.info(f"Model beta {lit_model.target.dist.action.beta}")

            logger.info(f"Executing for N samples {n_iter* batch_size}!")
            
            nip_repeat = []
            nmcmc_repeat = []
            if "nip" in eval_config.methods:
                for repeat_idx in range(eval_config.n_repeat):

                    stats_nip = lit_model.estimate_obs_nip(
                        batch_size=batch_size, n_iter=n_iter,ess_p=False
                    )
                    nip_repeat.append(stats_nip)

                if use_idx is not None:
                    stats_nip_list[use_idx] = nip_repeat
                else:
                    stats_nip_list.append(nip_repeat)

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


            eval_result.exact_sus = lit_model.sus_exact
            eval_result.nip = stats_nip_list
            eval_result.nmcmc = stats_nmcmc_list
            eval_result.n_samples = n_samples

            eval_result.save(task_dir)
