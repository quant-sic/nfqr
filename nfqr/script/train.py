import os
from argparse import ArgumentParser
from pathlib import Path
from venv import create

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.data.datasampler import FlowSampler
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import TrainConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils.misc import create_logger
import torch 

logger = create_logger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir


    train_config = TrainConfig.from_directory_for_task(exp_dir, task_id=int(os.environ["task_id"]),num_tasks=int(os.environ["num_tasks"]))
    log_dir = "task_{}".format(os.environ["task_id"])
    

    flow_model = LitFlow(**dict(train_config))

    if train_config.train_setup == "reverse":
        train_loader = FlowSampler(
            batch_size=train_config.trainer_config.batch_size,
            num_batches=train_config.trainer_config.num_batches,
            model=flow_model.model,
        )

    tb_logger = TensorBoardLogger(exp_dir/"logs", name=log_dir)

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
        auto_lr_find=True
    )

    logger.info("Train Config for task {}:\n {}".format(os.environ["task_id"],train_config))

    trainer.fit(model=flow_model, train_dataloaders=train_loader)

