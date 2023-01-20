from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pytorch_lightning as pl

from nfqr.utils import create_logger

log = create_logger(__name__)


def step_linear_increment_check(step: int, r: float = 0.5):
    """
    check if step is the next checkpoint step

    Args:
        step (int): current step
        num_chkpt (int): number of checkpoints
        r (float): ratio of steps between checkpoints (default: 0.5)
    """

    return np.round(r * step * (step + 1) / 2)


def step_frequency_check(step: int, f: int = 100, **kwargs):
    """
    check if step is a multiple of f

    Args:
        step (int): current step
        f (int): frequency
    """
    return step % f == 0


class CheckpointSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        dirpath: Path,
        prefix="step-checkpoint",
        use_modelcheckpoint_filename=False,
        mode: Literal["linear_increment", "frequency"] = "linear_increment",
        mode_kwargs: Any = {"r": 0.5},
        save_step_frequency: Optional[int] = None,
        number_of_saved_ckpts: int = 25,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """

        if mode == "linear_increment":
            self.save_steps = step_linear_increment_check(
                np.arange(1, number_of_saved_ckpts), **mode_kwargs
            )
            self.save_steps_check = lambda step, n_ckpt: step in self.save_steps

        elif mode == "frequency":
            self.save_steps_check = lambda step, n_ckpt: step_frequency_check(
                step, save_step_frequency
            )

        self.dirpath = dirpath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.number_of_saved_ckpts = 0

    def on_batch_end(self, trainer: pl.Trainer, _):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        # log.info(
        #     f"{global_step}, {self.number_of_saved_ckpts},{self.save_steps_check(global_step, self.number_of_saved_ckpts)}"
        # )
        if self.save_steps_check(global_step, self.number_of_saved_ckpts):
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"

            trainer.save_checkpoint(self.dirpath / filename)
            self.number_of_saved_ckpts += 1
