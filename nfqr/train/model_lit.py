from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Union,Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS,EVAL_DATALOADERS,STEP_OUTPUT,EPOCH_OUTPUT
from scipy import stats

from nfqr.eval.evaluation import (
    estimate_ess_p_nip,
    estimate_obs_nip,
    estimate_obs_nmcmc,
    get_ess_p_sampler,
)
from nfqr.normalizing_flows.flow import BareFlow, FlowConfig
from nfqr.normalizing_flows.loss.loss import LOSS_REGISTRY, LossConfig
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig, action
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.train.config import TrainerConfig
from nfqr.train.scheduler import SCHEDULER_REGISTRY, BetaScheduler, LossScheduler
from nfqr.utils import create_logger

logger = create_logger(__name__)


@dataclass
class Metrics:

    metrics_dict: Dict = field(default_factory=lambda: defaultdict(list))

    def add_batch_wise(
        self,
        _metrics_dict: Dict[str, Union[int, float]],
    ):

        for _key, _value in _metrics_dict.items():
            if isinstance(_value, torch.Tensor):
                _value = _value.item()
            self.metrics_dict[_key] += [_value]

    def last_slope(self, key: str, window_length: int):

        data = self.metrics_dict[key][-window_length:]
        slope = stats.linregress(np.arange(len(data)), data).slope

        slope_per_window = slope * len(data)
        return slope_per_window

    def last_mean(self, key: str, window_length: int):

        data = self.metrics_dict[key][-window_length:]
        return sum(data) / len(data)


class LitFlow(pl.LightningModule):
    def __init__(
        self,
        dim: List[int],
        flow_config: FlowConfig,
        observables: List[OBSERVABLE_REGISTRY.enum],
        action_config: ActionConfig,
        trainer_config: TrainerConfig,
        **kwargs,
    ) -> None:
        super().__init__()

        self.trainer_config = trainer_config
        self.action_config = action_config
        self.observables = observables
        self.dim = dim

        self.learning_rate = trainer_config.learning_rate
        self.model = BareFlow(**dict(flow_config))


    @cached_property
    def target(self):
        return TargetDensity.boltzmann_from_action(
                ACTION_REGISTRY[self.action_config.target_system][self.action_config.action_type](**dict(self.action_config.specific_action_config))
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

    def load_scheduler(self,config):
        _scheduler = SCHEDULER_REGISTRY[config.scheduler_type](
            **dict(config.specific_scheduler_config)
        )
        if "metrics" in dir(_scheduler):
            _scheduler.metrics = self.metrics
        if "target_action" in dir(_scheduler):
            _scheduler.target_action = self.target.dist.action

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

        _schedulers+=[self.loss_scheduler]

        if not all(len(set(list(filter(lambda _scheduler: isinstance(_scheduler,_scheduler_class),_schedulers))))<=1 for _scheduler_class in (BetaScheduler,LossScheduler)):
            raise RuntimeError("Only one scheduler instance per type is allowed")

        return _schedulers

    @property
    def sus_exact(self):
        return SusceptibilityExact(self.target.dist.action.beta, *self.dim).evaluate()

    @cached_property
    def sus_exact_final(self)->float:
        if not any(isinstance(_scheduler,BetaScheduler) for _scheduler in self.schedulers):
            return SusceptibilityExact(self.target.dist.action.beta, *self.dim).evaluate()
        else:
            beta_scheduler = filter(lambda _scheduler:isinstance(_scheduler,BetaScheduler),self.schedulers).__next__()
            return SusceptibilityExact(beta_scheduler.target_beta, *self.dim).evaluate()

    @cached_property
    def observables_fn(self):
        return {
            obs: OBSERVABLE_REGISTRY[self.action_config.target_system][obs]()
            for obs in self.observables
        }

    @cached_property
    def ess_p_sampler(self):
        if any(isinstance(_scheduler,BetaScheduler) for _scheduler in self.schedulers):
            logger.warning("Ess P sampler is not scheduled with beta. so ess p values will not be valid")

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
            for key,value in scheduler.log_stats.items():
                self.log(key,value)

        return {"loss": losses.mean()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def validation_step(self,batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        
        val_step_output={}
        batch_dict = self.val_losses[dataloader_idx].name_batch(batch)

        for obs_name, obs_fn in self.observables_fn.items():
            with torch.no_grad():
                val_step_output[obs_name] = obs_fn.evaluate(batch_dict["x_samples"])

        return val_step_output

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:

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


        if len(self.trainer.val_dataloaders)==1:
            outputs = [outputs]

        for dataloader_idx,val_steps_output in enumerate(outputs):
            val_steps_output_transposed=defaultdict(list)
            
            for output in val_steps_output:
                for key,val in output.items():
                    val_steps_output_transposed[key]+=[val] 

            for key,val in val_steps_output_transposed.items():
                if key in self.observables_fn.keys() and hasattr(self.observables_fn[key],"hist_bin_range"):
                    self.logger.experiment.add_histogram(tag=f"{dataloader_idx}_{type(self.trainer.val_dataloaders[dataloader_idx]).__name__}/{key}", values=torch.concat(val),global_step = self.global_step,bins = self.observables_fn[key].hist_bin_range(self.dim))


        # if "von_mises" in self.config.flow_config.base_dist_config.type:
        #     with torch.no_grad():
        #         self.log(
        #             "base_dist/concentration",
        #             self.model.base_distribution.constraint_transform(
        #                 self.model.base_distribution.concentration
        #             ),
        #         )


    def on_train_epoch_end(self) -> None:
        
        self.log("lr", self.learning_rate)

        self.log(
            "loss_epoch",
            self.metrics.last_mean("loss", self.trainer_config.train_num_batches),
        )

        self.log(
            "loss_std_epoch",
            self.metrics.last_mean("loss_std", self.trainer_config.train_num_batches),
        )

        return super().on_train_epoch_end()             

        
    def estimate_obs_nip(self, batch_size, n_iter):

        stats_nip = estimate_obs_nip(
            model=self.model,
            target=self.target,
            observables=self.observables,
            batch_size=batch_size,
            n_iter=n_iter,
        )

        stats_nip["ess_p"] = self.estimate_ess_p_nip(
            batch_size=self.trainer_config.batch_size_eval,
            n_iter=self.trainer_config.n_iter_eval,
        )

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

    def estimate_ess_p_nip(self, batch_size, n_iter):

        ess_p = estimate_ess_p_nip(
            model=self.model,
            data_sampler=self.ess_p_sampler,
            target=self.target,
            batch_size=batch_size,
            n_iter=n_iter,
        )

        return ess_p
