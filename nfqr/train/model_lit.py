from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)

from nfqr.eval.evaluation import (
    estimate_ess_p_nip,
    estimate_obs_nip,
    estimate_obs_nmcmc,
    get_ess_p_sampler,
)
from nfqr.normalizing_flows.flow import BareFlow, FlowConfig
from nfqr.normalizing_flows.loss.loss import LOSS_REGISTRY
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.train.config import TrainerConfig
from nfqr.train.metrics import Metrics
from nfqr.train.scheduler import (
    SCHEDULER_REGISTRY,
    BetaScheduler,
    LossScheduler,
    MaxFluctuationLRScheduler,
)
from nfqr.utils import create_logger

logger = create_logger(__name__)


class LitFlow(pl.LightningModule):
    def __init__(
        self,
        dim: List[int],
        flow_config: FlowConfig,
        observables: List[OBSERVABLE_REGISTRY.enum],
        action_config: ActionConfig,
        trainer_config: TrainerConfig,
        mode="train",
        **kwargs,
    ) -> None:
        super().__init__()

        self.trainer_config = trainer_config
        self.action_config = action_config
        self.observables = observables
        self.dim = dim

        self.model = BareFlow(**dict(flow_config))

        if mode == "eval":
            self.set_final_beta()
        elif mode == "train":
            if self.trainer_config is None:
                raise ValueError("Trainer Config None not allowed for mode == train!")

            self.learning_rate = trainer_config.learning_rate

    @cached_property
    def target(self):
        return TargetDensity.boltzmann_from_action(
            ACTION_REGISTRY[self.action_config.target_system][
                self.action_config.action_type
            ](**dict(self.action_config.specific_action_config))
        )

    @cached_property
    def metrics(self):
        return Metrics()

    @cached_property
    def train_losses(self):
        _losses = []
        for loss_config in self.trainer_config.loss_configs:
            loss = LOSS_REGISTRY[loss_config.loss_type](
                **dict(loss_config.specific_loss_config),
                batch_size=self.trainer_config.batch_size,
                num_batches=self.trainer_config.train_num_batches,
                model=self.model,
            )
            if "target" in dir(loss):
                loss.target = self.target

            _losses += [loss]

        return _losses

    @cached_property
    def val_losses(self):
        _losses = []
        for loss_config in self.trainer_config.loss_configs:
            loss = LOSS_REGISTRY[loss_config.loss_type](
                **dict(loss_config.specific_loss_config),
                batch_size=self.trainer_config.batch_size,
                num_batches=self.trainer_config.val_num_batches,
                model=self.model,
            )
            if "target" in dir(loss):
                loss.target = self.target

            _losses += [loss]

        return _losses

    def load_scheduler(self, config):
        _scheduler = SCHEDULER_REGISTRY[config.scheduler_type](
            **dict(config.specific_scheduler_config)
        )
        if "metrics" in dir(_scheduler):
            _scheduler.metrics = self.metrics
        if "target_action" in dir(_scheduler):
            _scheduler.target_action = self.target.dist.action
        if "model" in dir(_scheduler):
            _scheduler.model = self.model

        return _scheduler

    @cached_property
    def loss_scheduler(self):
        if self.trainer_config.loss_scheduler_config is None:
            return SCHEDULER_REGISTRY["default_loss"]()
        else:
            return self.load_scheduler(self.trainer_config.loss_scheduler_config)

    @cached_property
    def schedulers(self):
        _schedulers = []

        for scheduler_config in self.trainer_config.scheduler_configs:
            _schedulers += [self.load_scheduler(scheduler_config)]

        _schedulers += [self.loss_scheduler]

        if not all(
            len(
                set(
                    list(
                        filter(
                            lambda _scheduler: isinstance(_scheduler, _scheduler_class),
                            _schedulers,
                        )
                    )
                )
            )
            <= 1
            for _scheduler_class in (BetaScheduler, LossScheduler)
        ):
            raise RuntimeError("Only one scheduler instance per type is allowed")

        return _schedulers

    @property
    def sus_exact(self):
        return SusceptibilityExact(self.target.dist.action.beta, *self.dim).evaluate()

    @property
    def final_beta(self):
        if not any(
            isinstance(_scheduler, BetaScheduler) for _scheduler in self.schedulers
        ):
            return self.target.dist.action.beta
        else:
            # it is already checked that there is only one beta scheduler
            beta_scheduler = filter(
                lambda _scheduler: isinstance(_scheduler, BetaScheduler),
                self.schedulers,
            ).__next__()
            return beta_scheduler.target_beta

    @cached_property
    def sus_exact_final(self) -> float:
        return SusceptibilityExact(self.final_beta, *self.dim).evaluate()

    def set_final_beta(self):
        self.target.dist.action.beta = self.final_beta

    @cached_property
    def observables_fn(self):
        return {
            obs: OBSERVABLE_REGISTRY[self.action_config.target_system][obs]()
            for obs in self.observables
        }

    @cached_property
    def ess_p_sampler(self):
        if any(isinstance(_scheduler, BetaScheduler) for _scheduler in self.schedulers):
            logger.warning(
                "Ess P sampler is not scheduled with beta. so ess p values will not be valid"
            )

        return get_ess_p_sampler(
            dim=self.dim,
            action_config=self.action_config,
            batch_size=self.trainer_config.batch_size_eval,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        train_loaders = []
        for loss in self.train_losses:
            if hasattr(loss, "sampler"):
                train_loaders += [loss.sampler]
            else:
                raise ValueError("Unhandled sampler case")

        return train_loaders

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loaders = []
        for loss in self.val_losses:
            if hasattr(loss, "sampler"):
                val_loaders += [loss.sampler]
            else:
                raise ValueError("Unhandled sampler case")

        return val_loaders

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def training_step(self, batches, *args, **kwargs):

        losses_out_batch = []
        for loss, batch in zip(self.train_losses, batches):
            losses_out_batch += [loss.evaluate(batch)]

        metrics = self.loss_scheduler.evaluate(losses_out_batch)
        losses = metrics["loss_batch"]

        metrics_dict = {}
        for obs_name, obs_fn in self.observables_fn.items():
            with torch.no_grad():
                obs_values = obs_fn.evaluate(metrics["x_samples"])
                metrics_dict[obs_name] = obs_values.mean()

        metrics_dict.update({"loss": losses.mean(), "loss_std": losses.std()})
        self.metrics.add_batch_wise(metrics_dict)

        for key, value in metrics_dict.items():
            self.log(key, value)

        for scheduler in self.schedulers:
            scheduler.step()
            if hasattr(scheduler,"log_stats"):
                for key, value in scheduler.log_stats.items():
                    self.log(key, value)

        self.log_model_pars()

        return {"loss": losses.mean()}

    def configure_optimizers(self):

        configuration_dict = {}
        if self.trainer_config.optimizer == "Adam":
            configuration_dict["optimizer"] = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate
            )
        else:
            raise ValueError(
                "Unknown optimizer type {}".format(self.trainer_config.optimizer)
            )

        lr_scheduler_dict = self.trainer_config.lr_scheduler
        if lr_scheduler_dict is None:
            pass
        elif lr_scheduler_dict["type"] == "reduce_on_plateau":
            configuration_dict["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=configuration_dict["optimizer"],
                    patience=lr_scheduler_dict.get("patience", 10),
                    factor=lr_scheduler_dict.get("factor", 0.9),
                    min_lr=lr_scheduler_dict.get("min_lr", 5e-5),
                ),
                "interval": "epoch",
                "monitor": "loss",
            }
        elif lr_scheduler_dict["type"] == "max_fluctuations":
            lr_scheduler = MaxFluctuationLRScheduler(
                    optimizer=configuration_dict["optimizer"],
                    max_fluctuation_base=lr_scheduler_dict.get(
                        "max_fluctuation_base", 0.05
                    ),
                    final_max_fluctuations=lr_scheduler_dict.get(
                        "final_max_fluctuations", None
                    ),
                    n_steps=lr_scheduler_dict.get("n_steps", None),
                    max_fluctuation_step=lr_scheduler_dict.get(
                        "max_fluctuation_step", 0.001
                    ),
                    cooldown_steps=lr_scheduler_dict.get("patience", 75),
                    metric_window_length=lr_scheduler_dict.get(
                        "metric_window_length", 100
                    ),
                    change_rate=lr_scheduler_dict.get("factor", 0.9),
                    min_lr=lr_scheduler_dict.get("min_lr", 5e-5),
                )
            lr_scheduler.metrics = self.metrics
            configuration_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
        else:
            raise ValueError(
                "Unknown lr_scheduler type {}".format(
                    self.trainer_config.lr_scheduler["type"]
                )
            )

        return configuration_dict

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        
        if self.current_epoch > self.trainer_config.lr_scheduler.get("initial_waiting_epochs",0):
            if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics=metric)
            else:
                scheduler.step()

    def log_all_values_in_stats_dict(
        self, node: Union[Dict, int, float], str_path_to_node: str
    ):
        """
        Recursively logs all leaves in stats dict
        """
        if isinstance(node, dict):
            for key, _node in node.items():
                self.log_all_values_in_stats_dict(_node, f"{str_path_to_node}/{key}")

        elif isinstance(node, (int, float, torch.Tensor)):

            if isinstance(node, int):
                node = float(node)
            elif isinstance(node, torch.Tensor):
                node = node.to(torch.float32)

            self.log(str_path_to_node, node)

            if (
                "Chi_t" in str_path_to_node.split("/")[-2]
                and "mean" in str_path_to_node.split("/")[-1]
            ):
                self.log(
                    f"{str_path_to_node}/abs_diff_to_exact", abs(node - self.sus_exact)
                )
                self.log(f"{str_path_to_node}/sus_exact", self.sus_exact)

        else:
            raise ValueError(
                f"Unknown node type in stats dict {type(node)} for node {node}"
            )

    def log_model_pars(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "logging_parameters"):
                self.log_all_values_in_stats_dict(
                    node=module.logging_parameters,
                    str_path_to_node="flow/" + name + "_" + type(module).__name__,
                )

    def validation_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> Optional[STEP_OUTPUT]:

        # self.val_losses[dataloader_idx].evaluate(batch)

        val_step_output = {}

        batch_dict = self.val_losses[dataloader_idx].name_batch(batch)

        for obs_name, obs_fn in self.observables_fn.items():
            with torch.no_grad():
                val_step_output[obs_name] = obs_fn.evaluate(batch_dict["x_samples"])

        return val_step_output

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:

        stats_nmcmc = self.estimate_obs_nmcmc(
            batch_size=self.trainer_config.batch_size_eval,
            n_iter=self.trainer_config.n_iter_eval,
        )

        stats_nip = self.estimate_obs_nip(
            batch_size=self.trainer_config.batch_size_eval,
            n_iter=self.trainer_config.n_iter_eval,
        )

        for sampler, _stats in zip(("nip", "nmcmc"), (stats_nip, stats_nmcmc)):
            self.log_all_values_in_stats_dict(_stats, sampler)

        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

        for dataloader_idx, val_steps_output in enumerate(outputs):
            val_steps_output_transposed = defaultdict(list)

            for output in val_steps_output:
                for key, val in output.items():
                    val_steps_output_transposed[key] += [val]

            for key, val in val_steps_output_transposed.items():
                if key in self.observables_fn.keys() and hasattr(
                    self.observables_fn[key], "hist_bin_range"
                ):
                    self.logger.experiment.add_histogram(
                        tag=f"{dataloader_idx}_{type(self.trainer.val_dataloaders[dataloader_idx]).__name__}/{key}",
                        values=torch.concat(val),
                        global_step=self.global_step,
                        bins=self.observables_fn[key].hist_bin_range(self.dim),
                    )

        # if "von_mises" in self.config.flow_config.base_dist_config.type:
        #     with torch.no_grad():
        #         self.log(
        #             "base_dist/concentration",
        #             self.model.base_distribution.constraint_transform(
        #                 self.model.base_distribution.concentration
        #             ),
        #         )

    def on_train_epoch_end(self) -> None:

        self.log(
            "loss_epoch",
            self.metrics.last_mean("loss", self.trainer_config.train_num_batches),
        )

        self.log(
            "loss_std_epoch",
            self.metrics.last_mean("loss_std", self.trainer_config.train_num_batches),
        )

        for scheduler in [self.lr_schedulers()]:
            if hasattr(scheduler,"log_stats"):
                for key, value in scheduler.log_stats.items():
                    self.log(key, value)

        return super().on_train_epoch_end()

    def estimate_obs_nip(self, batch_size, n_iter):

        stats_nip = estimate_obs_nip(
            model=self.model,
            target=self.target,
            observables=self.observables,
            batch_size=batch_size,
            n_iter=n_iter,
        )

        stats_nip["ess_p"] = self.estimate_ess_p_nip()

        return stats_nip

    def estimate_obs_nmcmc(self, batch_size, n_iter):

        stats_nmcmc = estimate_obs_nmcmc(
            model=self.model,
            observables=self.observables,
            target=self.target,
            trove_size=batch_size,
            n_steps=n_iter * batch_size,
        )

        return stats_nmcmc

    def estimate_ess_p_nip(self):

        ess_p = estimate_ess_p_nip(
            model=self.model,
            data_sampler=self.ess_p_sampler,
            target=self.target,
            batch_size=self.ess_p_sampler.batch_size,
            n_iter=int(len(self.ess_p_sampler.dataset) / self.ess_p_sampler.batch_size),
        )

        return ess_p
