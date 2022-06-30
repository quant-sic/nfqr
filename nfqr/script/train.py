import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    parser.add_argument("--skip_done",type=bool,default=True)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
        num_tasks=int(os.environ["num_tasks"]),
    )
    log_dir = "task_{}".format(os.environ["task_id"])

    if not len(list(((exp_dir / "logs")/log_dir).glob("*")))==0 and args.skip_done:
        logger.info("Task already executed. Aborting!")

    else:

        seed_everything(True,workers=True)

        flow_model = LitFlow(**dict(train_config))

        tb_logger = TensorBoardLogger(exp_dir / "logs", name=log_dir)

        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="loss_std")
        trainer = Trainer(
            **train_config.trainer_config.dict(
                include={
                    "max_epochs",
                    "log_every_n_steps",
                    "accumulate_grad_batches",
                }
            ),
            logger=tb_logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            auto_lr_find=train_config.trainer_config.auto_lr_find,
            default_root_dir=(exp_dir / "trainer")/log_dir
        )

        logger.info(
            "Train Config for task {}:\n {}".format(os.environ["task_id"], train_config)
        )

        if train_config.trainer_config.auto_lr_find:
            trainer.tune(model=flow_model,lr_find_kwargs={
                    "min_lr": 1e-8,
                    "max_lr": 5e-3,
                    "num_training": 250,
                    "mode": "exponential",
                    "early_stop_threshold": 5.0,
                    "update_attr": True,
                })
            if flow_model.learning_rate is None:
                flow_model.learning_rate = train_config.trainer_config.learning_rate

        trainer.fit(model=flow_model)
