import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def train_flow_model_tune(config, exp_dir):

    exp_dir = EXPERIMENTS_DIR / exp_dir

    assert int(os.environ["num_tasks"]) == 1
    assert int(os.environ["task_id"]) == 0

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"]),
        num_tasks=int(os.environ["num_tasks"]),
        tune_config=config,
    )

    log_dir = "task_{}".format(os.environ["task_id"])

    flow_model = LitFlow(**dict(train_config))

    tb_logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")

    tune_report_callback = TuneReportCallback(
        {"loss": "loss", "mean_accuracy": "nip/ess_p/0-1/ess_p"},
        on="validation_end",
    )

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
        default_root_dir=(exp_dir / "trainer") / log_dir,
        callbacks=[tune_report_callback]
    )

    trainer.fit(flow_model)


def tune_flow_model_asha(exp_dir):

    config = {"num_layers": tune.choice([2, 3, 8])}

    num_epochs = 1000
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_flow_model_tune, exp_dir=exp_dir
    )
    resources_per_trial = {"cpu": 2, "gpu": 1}

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_flow_model_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    train_flow_model_tune(exp_dir=args.exp_dir)