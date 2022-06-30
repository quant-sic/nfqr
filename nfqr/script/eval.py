from argparse import ArgumentParser
from pathlib import Path
import os

from nfqr.eval.evaluation import EvalConfig,EvalResult
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from tqdm import tqdm

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    train_config = LitModelConfig.from_directory_for_task(exp_dir, task_id=int(os.environ["task_id"]),num_tasks=int(os.environ["num_tasks"]))
    eval_config = EvalConfig.from_directory(exp_dir)

    log_dir = "task_{}".format(os.environ["task_id"])
 
    pbar = tqdm((exp_dir / f"logs/{log_dir}").glob("**/*.ckpt"))
    for model_ckpt_path in pbar:

        pbar.set_description("Evaluation for task {} and model {}".format(os.environ["task_id"],model_ckpt_path.stem))

        task_dir = exp_dir/f"eval/{log_dir}/{model_ckpt_path.stem}"

        lit_model = LitFlow.load_from_checkpoint(model_ckpt_path, **dict(train_config))

        eval_result = EvalResult(observables=eval_config.observables,n_samples=[b*n for (b,n) in zip(eval_config.batch_size,eval_config.n_iter)])

        stats_nip_list = []
        stats_nmcmc_list = []

        for n_iter,batch_size in zip(eval_config.n_iter,eval_config.batch_size):
            
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

        eval_result.exact_sus = lit_model.sus_exact_final
        eval_result.nip = stats_nip_list
        eval_result.nmcmc = stats_nmcmc_list
        
        eval_result.save(task_dir)
