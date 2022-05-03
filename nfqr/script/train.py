import os
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.data.datasampler import FlowSampler
from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import TrainConfig
from nfqr.train.model_lit import LitFlow

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.exp_dir

    if "SGE_TASK_ID" not in os.environ:
        train_config = TrainConfig.from_directory_for_task(
            exp_dir, task_id=os.environ["SGE_TASK_ID"]
        )
    else:
        train_config = TrainConfig.from_directory(exp_dir)

    flow_model = LitFlow(**dict(train_config))

    if train_config.train_setup == "reverse":
        train_loader = FlowSampler(
            batch_size=train_config.batch_size,
            num_batches=train_config.num_batches,
            model=flow_model.model,
        )

    logger = TensorBoardLogger(exp_dir, name="logs")
    trainer = Trainer(max_epochs=train_config.max_epochs, logger=logger, gpus=0)

    trainer.fit(model=flow_model, train_dataloaders=train_loader)

    trainer.save_checkpoint(exp_dir / "model.ckpt")
