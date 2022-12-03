import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from nfqr.globals import EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils.misc import create_logger
import numpy as np
from nfqr.utils import setup_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nfqr.train.model_lit import LitFlow
from nfqr.train.config import LitModelConfig
import re
from tqdm import tqdm 

logger = create_logger(__name__)


if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR/args.exp_dir

    events_file_path = exp_dir.glob("**/event*").__next__()

    acc = EventAccumulator(str(events_file_path)).Reload()

    _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]

    train_config = LitModelConfig.from_directory_for_task(exp_dir,task_id=0,num_tasks=1)
    trainer_config = train_config.trainer_configs[0]
    steps,saved_regular_model_paths = zip(*list((int(re.search("step=([0-9]*).",p.name).groups()[0]),p) for p in exp_dir.glob("**/regular/*.ckpt")))
    
    plot_steps = np.argwhere((step_nums[:,None]>np.array(steps)[None,:]).cumsum(axis=0)==1)[:,0]
    beta_values = vals[plot_steps]

    for model_path,beta in tqdm(zip(saved_regular_model_paths,beta_values),desc="Evaluating regular models"):

        train_config.action_config.specific_action_config.beta = beta
        model = LitFlow.load_from_checkpoint(model_path,**dict(train_config),trainer_config=trainer_config)

        logger.info(beta)
        logger.info(model.target.dist.action.beta)
        logger.info(model.estimate_ess_p_nip())