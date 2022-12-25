import os
import re
import shutil
import timeit
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import optim

from nfqr.config import BaseConfig
from nfqr.data import (ConditionConfig, MCMCConfig, PSampler,
                       TrajectorySamplerConfig)
from nfqr.globals import EXPERIMENTS_DIR, TMP_DIR
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.mcmc.nmcmc import NeuralMCMC
from nfqr.nip import (NeuralImportanceSampler,
                      calc_ess_p_from_unnormalized_log_weights,
                      calc_ess_q_from_unnormalized_log_weights)
from nfqr.target_systems import OBSERVABLE_REGISTRY
from nfqr.target_systems.rotor import RotorTrajectorySamplerConfig
from nfqr.train.config import LitModelConfig
from nfqr.train.model_lit import LitFlow
from nfqr.utils import create_logger, setup_env
from nfqr.utils.misc import create_logger
from nfqr.eval.evaluation import get_tmp_path_from_name_and_environ
from collections import defaultdict
import json
from nfqr.data.datasampler import ExtendedDLIterator

logger = create_logger(__name__)


def train_routine(dl_iter,flow_model,optimizer):

    optimizer.zero_grad()

    batch = dl_iter.__next__()
    losses_out_batch = []
    for loss, batch in zip(flow_model.train_losses, (batch,)):
        losses_out_batch += [loss.evaluate(batch)]

    metrics = flow_model.loss_scheduler.evaluate(losses_out_batch)
    losses = metrics["loss_batch"]
    
    losses.mean().backward()
    optimizer.step()


def run_no_grad(sampler):
    with torch.no_grad():
        sampler.run()


def iterate_config_models(exp_dir, model_ckpt_path):

    train_config = LitModelConfig.from_directory_for_task(
        exp_dir,
        task_id=int(os.environ["task_id"])    
        )

    for idx, trainer_config in enumerate(train_config.trainer_configs):

        lit_model_config = dict(train_config)
        lit_model_config.update({"trainer_config": trainer_config})

        if train_config.continue_beta is not None and train_config.continuation_exp is not None:
            logger.info(
                f"Continuing training for beta={train_config.continue_beta} for experiment = {train_config.continuation_exp}")

            events_file_path = (
                EXPERIMENTS_DIR / train_config.continuation_exp).glob("**/events*").__next__()

            acc = EventAccumulator(str(events_file_path)).Reload()
            _, step_nums, vals = [np.array(e)
                                  for e in zip(*acc.Scalars("beta"))]

            steps, saved_regular_model_paths = zip(*list((int(re.search("step=([0-9]*).", p.name).groups(
            )[0]), p) for p in (EXPERIMENTS_DIR / train_config.continuation_exp).glob("**/regular/*.ckpt")))

            plot_steps_ = np.argwhere((step_nums[:, None] > np.array(steps)[
                None, :]).cumsum(axis=0) == 1)

            plot_steps = plot_steps_[np.argsort(plot_steps_[:, 1]), 0]

            beta_values = vals[plot_steps]

            #choose model which is closest but needs to be larger beta
            step_diffs = beta_values-train_config.continue_beta
            step_diffs[step_diffs<0] = np.inf
            closest_idx = np.argmin(step_diffs)
            
            beta_ckpt_path = saved_regular_model_paths[closest_idx]
            lit_model_config["action_config"].specific_action_config.beta = train_config.continue_beta

            logger.info(f"Using saved checkpoint {beta_ckpt_path}")
            flow_model = LitFlow.load_from_checkpoint(
                beta_ckpt_path, **lit_model_config)

        elif train_config.continue_model is not None and train_config.continuation_exp is not None:

            continue_model_path = (EXPERIMENTS_DIR/train_config.continuation_exp).glob(f"**/{train_config.continue_model}*").__next__()
            events_file_path = (
                EXPERIMENTS_DIR / train_config.continuation_exp).glob("**/events*").__next__()

            acc = EventAccumulator(str(events_file_path)).Reload()
            _, step_nums, vals = [np.array(e) for e in zip(*acc.Scalars("beta"))]

            step = int(re.search("step=([0-9]*).",continue_model_path.name).groups()[0])
            
            plot_step = np.argwhere((step_nums>=(step-1)).cumsum(axis=0)==1)[0][0]
            beta = vals[plot_step]

            lit_model_config["action_config"].specific_action_config.beta = beta
            logger.info(f"Setting beta to {beta}")
            
            flow_model = LitFlow.load_from_checkpoint(
                continue_model_path, **lit_model_config)

        elif idx == 0:
            flow_model = LitFlow(**lit_model_config)
        else:
            flow_model = LitFlow.load_from_checkpoint(
                model_ckpt_path, **lit_model_config
            )

        yield lit_model_config, trainer_config, flow_model


def train_flow_model(exp_dir):

    exp_dir = EXPERIMENTS_DIR / exp_dir

    log_dir = "task_{}/timed".format(os.environ["task_id"])

    seed_everything(42, workers=True)
    model_ckpt_path = ((exp_dir / "logs") / log_dir) / "model.ckpt"

    for idx, (lit_model_config, trainer_config, flow_model) in enumerate(iterate_config_models(exp_dir, model_ckpt_path)):

        logger.info(
            "Task {}: Interval {} with: \n\n {} \n\n".format(
                os.environ["task_id"], idx, lit_model_config
            )
        )

        tb_logger = TensorBoardLogger(
            exp_dir / "logs", name=log_dir, sub_dir=f"interval_{idx}", version=0
        )

        # callbacks = [
        #     ModelCheckpoint(dirpath=tb_logger.log_dir + "/checkpoints/max_ess_p",
        #                     auto_insert_metric_name=True, save_top_k=3, monitor="nip/ess_p/0-1/ess_p", mode="max"),
        #     ModelCheckpoint(dirpath=tb_logger.log_dir + "/checkpoints/regular",
        #                     filename="latest-{epoch}-{step}", save_top_k=-1, monitor="step", mode="max", every_n_epochs=25),
        #     LearningRateMonitor(logging_interval='step')
        # ]

        #accelerator = ("mps","gpu","cpu")[np.argwhere((torch.backends.mps.is_available(),torch.cuda.is_available(),True)).min()]
        accelerator = ("gpu", "cpu")[np.argwhere(
            (torch.cuda.is_available(), True)).min()]

        trainer = Trainer(
            **trainer_config.dict(
                include={
                    "log_every_n_steps",
                    "accumulate_grad_batches",
                    "gradient_clip_val",
                    "gradient_clip_algorithm",
                    "track_grad_norm",
                }
            ),
            max_epochs=0,
            num_sanity_val_steps=0,
            logger=tb_logger,
            accelerator=accelerator,
            devices=1,
            auto_lr_find=trainer_config.auto_lr_find,
            default_root_dir=(exp_dir / "trainer") / log_dir,
            # callbacks=callbacks
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


        flow_model.automatic_optimization=False
        
        accelerator = ("cuda", "cpu")[np.argwhere(
            (torch.cuda.is_available(), True)).min()]
        flow_model.to(accelerator)
        optimizer = optim.Adam(params=flow_model.parameters())

        res_dict = defaultdict(dict)
        n_iter=5
        repeat=5

        # time training step
        train_data_loaders = flow_model.train_dataloader()
        for dl_idx,dl in enumerate(train_data_loaders):
            
            dl_iter = ExtendedDLIterator(dl)            

            # time this
            timer = timeit.Timer(stmt='train_routine(dl_iter=dl_iter,flow_model=flow_model,optimizer=optimizer)', setup='pass',globals={**globals(),**locals()})
            time_res = timer.repeat(repeat=repeat,number=3)
            res_dict["train"][dl_idx] = (n_iter,time_res)
        

        # float evaluations
        flow_model.eval()
        
        for precision in ("double","float"):

            if precision =="double":
                flow_model.model.double()
            else:
                flow_model.model.float()

            rec_tmp = get_tmp_path_from_name_and_environ("estimate_obs_nip")
            nip_sampler = NeuralImportanceSampler(
                model=flow_model.model,
                target=flow_model.target,
                n_iter=n_iter,
                batch_size=10000,
                observables=flow_model.observables,
                out_dir=rec_tmp,
            )
            timer = timeit.Timer(stmt='run_no_grad(sampler=nip_sampler)', setup='pass',globals={**globals(),**locals()})
            time_res = timer.repeat(repeat=repeat,number=3)
            res_dict["nip_q"][precision] = (n_iter,time_res)

            shutil.rmtree(rec_tmp)


            rec_tmp = get_tmp_path_from_name_and_environ("estimate_nip_ess_p")
            nip_sampler = NeuralImportanceSampler(
                model=flow_model.model,
                target=flow_model.target,
                n_iter=n_iter,
                batch_size=10000,
                observables=[],
                out_dir=rec_tmp,
                mode="p",
                sampler=flow_model.ess_p_sampler,
            )
            timer = timeit.Timer(stmt='run_no_grad(sampler=nip_sampler)', setup='pass',globals={**globals(),**locals()})
            time_res = timer.repeat(repeat=repeat,number=3)
            res_dict["nip_p"][precision] = (n_iter,time_res)

            shutil.rmtree(rec_tmp)


            rec_tmp = get_tmp_path_from_name_and_environ("estimate_obs_nmcmc")
            nmcmc = NeuralMCMC(
                model=flow_model.model,
                target=flow_model.target,
                trove_size=10000,
                n_steps=n_iter*10000,
                observables=flow_model.observables,
                out_dir=rec_tmp,
            )
            timer = timeit.Timer(stmt='run_no_grad(sampler=nmcmc)', setup='pass',globals={**globals(),**locals()})
            time_res = timer.repeat(repeat=repeat,number=3)
            res_dict["nmcmc"][precision] = (n_iter,time_res)

            shutil.rmtree(rec_tmp)

        save_dir = ((exp_dir / "timed") / log_dir )/ "task_{}".format(os.environ["task_id"])
        save_dir.mkdir(exist_ok=True,parents=True)
        with open(save_dir/"result.json","w") as file:
            json.dump(res_dict,file)



if __name__ == "__main__":

    setup_env()

    parser = ArgumentParser()

    parser.add_argument("--exp_dir", type=Path)
    args = parser.parse_args()

    logger.info(f"Timing {args.exp_dir}")

    train_flow_model(exp_dir=args.exp_dir)
