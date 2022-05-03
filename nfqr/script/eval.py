import argparse
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.eval.config import EvalConfig
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import TrainConfig
from nfqr.train.model_lit import LitFlow

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)

    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--n_iter", type=int, default=20)

    parser.add_argument("--methods", type=str)
    # parser.add_argument("--observables", type=str)

    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    train_config = TrainConfig.from_directory(exp_dir)

    model_ckpt_path = exp_dir / "model.ckpt"

    lit_model = LitFlow.load_from_checkpoint(model_ckpt_path, **dict(train_config))

    eval_config = EvalConfig()

    if "nip" in args.methods:
        stats_nip = lit_model.estimate_obs_nip(
            batch_size=args.batch_size, n_iter=args.n_iter
        )
        eval_config.nip = stats_nip

    if "nmcmc" in args.methods:
        stats_nmcmc = lit_model.estimate_obs_nmcmc(
            batch_size=args.batch_size, n_iter=args.n_iter
        )
        eval_config.nmcmc = stats_nmcmc

    eval_config.exact_sus = lit_model.sus_exact

    eval_config.save(exp_dir)
