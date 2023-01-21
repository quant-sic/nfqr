import os
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.train.utils import CheckpointSteps
from nfqr.utils import setup_env
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def iterate_config_models(exp_dir, model_ckpt_path):

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
    )

    for idx, trainer_config in enumerate(train_config.trainer_configs):

        lit_model_config = dict(train_config)
        lit_model_config.update({"trainer_config": trainer_config})

        if (
            train_config.continue_beta is not None
            and train_config.continuation_exp is not None
        ):
            logger.info(
                f"Continuing training for beta={train_config.continue_beta} for experiment = {train_config.continuation_exp}"
            )

            events_file_path = (
                (EXPERIMENTS_DIR / train_config.continuation_exp)
                .glob("**/events*")
                .__next__()
            )

            acc = EventAccumulator(str(events_file_path)).Reload()
            _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]

            steps, saved_regular_model_paths = zip(
                *list(
                    (int(re.search("step=([0-9]*).", p.name).groups()[0]), p)
                    for p in (EXPERIMENTS_DIR / train_config.continuation_exp).glob(
                        "**/regular/*.ckpt"
                    )
                )
            )

            plot_steps_ = np.argwhere(
                (step_nums[:, None] > np.array(steps)[None, :]).cumsum(axis=0) == 1
            )

            plot_steps = plot_steps_[np.argsort(plot_steps_[:, 1]), 0]

            beta_values = vals[plot_steps]

            # choose model which is closest but needs to be larger beta
            step_diffs = beta_values - train_config.continue_beta
            step_diffs[step_diffs < 0] = np.inf
            closest_idx = np.argmin(step_diffs)

            beta_ckpt_path = saved_regular_model_paths[closest_idx]
            lit_model_config[
                "action_config"
            ].specific_action_config.beta = train_config.continue_beta

            logger.info(f"Using saved checkpoint {beta_ckpt_path}")
            flow_model = LitFlow.load_from_checkpoint(
                beta_ckpt_path, **lit_model_config
            )

        elif (
            (train_config.continue_model is not None)
            and (train_config.continuation_exp is not None)
            and ("beta_scheduled" in str(train_config.continuation_exp))
        ):

            continue_model_path = (
                (EXPERIMENTS_DIR / train_config.continuation_exp)
                .glob(f"**/{train_config.continue_model}*")
                .__next__()
            )
            events_file_path = (
                (EXPERIMENTS_DIR / train_config.continuation_exp)
                .glob("**/events*")
                .__next__()
            )

            acc = EventAccumulator(str(events_file_path)).Reload()
            _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]

            step = int(
                re.search("step=([0-9]*).", continue_model_path.name).groups()[0]
            )

            plot_step = np.argwhere((step_nums >= (step - 1)).cumsum(axis=0) == 1)[0][0]
            beta = vals[plot_step]

            lit_model_config["action_config"].specific_action_config.beta = beta
            logger.info(f"Setting beta to {beta}")

            flow_model = LitFlow.load_from_checkpoint(
                continue_model_path, **lit_model_config
            )

        elif idx == 0:
            flow_model = LitFlow(**lit_model_config)
        else:
            flow_model = LitFlow.load_from_checkpoint(
                model_ckpt_path, **lit_model_config
            )

        yield lit_model_config, trainer_config, flow_model


def train_flow_model(exp_dir, skip_done=True):

    exp_dir = EXPERIMENTS_DIR / exp_dir

    log_dir = "task_{}".format(os.environ["task_id"])

    if not len(list(((exp_dir / "logs") / log_dir).glob("*"))) == 0 and skip_done:
        logger.info("Task already executed. Aborting!")

    else:

        model_ckpt_path = ((exp_dir / "logs") / log_dir) / "model.ckpt"

        for idx, (lit_model_config, trainer_config, flow_model) in enumerate(
            iterate_config_models(exp_dir, model_ckpt_path)
        ):

            if trainer_config.reseed_random:
                seed_everything(trainer_config.random_seed, workers=True)
            else:
                seed_everything(42, workers=True)


            logger.info(
                "Task {}: Interval {} with: \n\n {} \n\n".format(
                    os.environ["task_id"], idx, lit_model_config
                )
            )

            tb_logger = TensorBoardLogger(
                exp_dir / "logs", name=log_dir, sub_dir=f"interval_{idx}", version=0
            )

            callbacks = [
                LearningRateMonitor(logging_interval="step"),
            ]
            if trainer_config.eval_ess_p:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=tb_logger.log_dir + "/checkpoints/max_ess_p",
                        auto_insert_metric_name=True,
                        save_top_k=3,
                        monitor="nip/ess_p/0-1/ess_p",
                        mode="max",
                    )
                )
            if trainer_config.save_regular:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=tb_logger.log_dir + "/checkpoints/regular",
                        filename="latest-{epoch}-{step}",
                        save_top_k=-1,
                        monitor="step",
                        mode="max",
                        every_n_epochs=trainer_config.save_every_n_epochs,
                    )
                )
            if trainer_config.save_steps:
                callbacks.append(
                    CheckpointSteps(
                        dirpath=Path(tb_logger.log_dir + "/checkpoints/steps"),
                        mode_kwargs={"r": trainer_config.save_steps_ratio},
                        number_of_saved_ckpts=trainer_config.number_of_saved_ckpts_steps_ratio,
                    )
                )

            # accelerator = ("mps","gpu","cpu")[np.argwhere((torch.backends.mps.is_available(),torch.cuda.is_available(),True)).min()]
            accelerator = ("gpu", "cpu")[
                np.argwhere((torch.cuda.is_available(), True)).min()
            ]

            trainer = Trainer(
                **trainer_config.dict(
                    include={
                        "max_epochs",
                        "log_every_n_steps",
                        "accumulate_grad_batches",
                        "gradient_clip_val",
                        "gradient_clip_algorithm",
                        "track_grad_norm",
                    }
                ),
                num_sanity_val_steps=1,
                logger=tb_logger,
                accelerator=accelerator,
                devices=1,
                auto_lr_find=trainer_config.auto_lr_find,
                default_root_dir=(exp_dir / "trainer") / log_dir,
                callbacks=callbacks,
            )

            if trainer_config.auto_lr_find:
                trainer.tune(
                    model=flow_model,
                    lr_find_kwargs={
                        "min_lr": 1e-8,
                        "max_lr": 5e-3,
                        "num_training": 250,
                        "mode": "exponential",
                        "early_stop_threshold": 5.0,
                        "update_attr": True,
                    },
                )
                if flow_model.learning_rate is None:
                    flow_model.learning_rate = trainer_config.learning_rate

            trainer.fit(model=flow_model)
            trainer.save_checkpoint(model_ckpt_path)


if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--skip_done", type=bool, default=True)
    args = parser.parse_args()

    train_flow_model(exp_dir=args.exp_dir, skip_done=args.skip_done)
