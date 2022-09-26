import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils.misc import create_logger
import numpy as np
from nfqr.utils import setup_env

logger = create_logger(__name__)


def train_flow_model(exp_dir, skip_done=True):

    exp_dir = EXPERIMENTS_DIR / exp_dir

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
        num_tasks=int(os.environ["num_tasks"]),
    )

    log_dir = "task_{}".format(os.environ["task_id"])

    if not len(list(((exp_dir / "logs") / log_dir).glob("*"))) == 0 and skip_done:
        logger.info("Task already executed. Aborting!")

    else:

        seed_everything(42, workers=True)
        model_ckpt_path = ((exp_dir / "logs") / log_dir) / "model.ckpt"

        for idx, trainer_config in enumerate(train_config.trainer_configs):

            lit_model_config = dict(train_config)
            lit_model_config.update({"trainer_config": trainer_config})

            logger.info(
                "Task {}: Interval {} with: \n\n {} \n\n".format(
                    os.environ["task_id"], idx, lit_model_config
                )
            )

            tb_logger = TensorBoardLogger(
                exp_dir / "logs", name=log_dir, sub_dir=f"interval_{idx}",version=0
            )

            if idx == 0:
                flow_model = LitFlow(**lit_model_config)
            else:
                flow_model = LitFlow.load_from_checkpoint(
                    model_ckpt_path, **lit_model_config
                )

            callbacks = [LearningRateMonitor()]


            #accelerator = ("mps","gpu","cpu")[np.argwhere((torch.backends.mps.is_available(),torch.cuda.is_available(),True)).min()]
            accelerator = ("gpu","cpu")[np.argwhere((torch.cuda.is_available(),True)).min()]

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
            #trainer.save_checkpoint(model_ckpt_path)


if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--skip_done", type=bool, default=True)
    args = parser.parse_args()

    train_flow_model(exp_dir=args.exp_dir, skip_done=args.skip_done)
