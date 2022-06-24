from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from scipy import stats

from nfqr.data.datasampler import FlowSampler
from nfqr.eval.evaluation import (
    estimate_ess_p_nip,
    estimate_obs_nip,
    estimate_obs_nmcmc,
    get_ess_p_sampler,
)
from nfqr.normalizing_flows.flow import BareFlow, FlowConfig
from nfqr.normalizing_flows.loss.loss import LOSS_REGISTRY, LossConfig, ReverseKL
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.train.config import TrainerConfig
from nfqr.train.scheduler import SCHEDULER_REGISTRY, BetaScheduler, BetaSchedulerConfig
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
        target_system: ACTION_REGISTRY.enum,
        action: ACTION_REGISTRY.enum,
        observables: List[OBSERVABLE_REGISTRY.enum],
        action_config: ActionConfig,
        trainer_config: TrainerConfig,
        loss_configs=List[LossConfig],
        learning_rate=0.001,
        **kwargs,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate

        self.model = BareFlow(**dict(flow_config))

        self.target = TargetDensity.boltzmann_from_action(
            ACTION_REGISTRY[target_system][action](**dict(action_config))
        )

        self.observables = observables
        self.target_system = target_system

        # if train_setup == "reverse":
        #     self.training_step = self._training_step_reverse
        #     self.train_dataloader = partial(
        #         self._train_dataloader_reverse,
        #         batch_size=trainer_config.batch_size,
        #         num_batches=trainer_config.num_batches,
        #     )

        # if train_setup == "forward":
        #     self.training_step = self._training_step_forward
        #     self.train_dataloader = partial(
        #         self._train_dataloader_forward,
        #         batch_size=trainer_config.batch_size,
        #         num_batches=trainer_config.num_batches,
        #     )
        #     assert (
        #         p_sampler_config is not None
        #     ), "P sampler config cannot be None in forward mode"
        #     self.p_sampler_config = p_sampler_config

        self.trainer_config = trainer_config
        self.dim = dim

        self.metrics = Metrics()

        self.loss_configs = loss_configs

    @cached_property
    def losses(self):
        _losses = []
        for loss_config in self.loss_configs:
            loss = LOSS_REGISTRY[loss_config.loss_type](
                **dict(loss_config.specific_loss_config),batch_size = self.trainer_config.batch_size,num_batches=self.trainer_config.num_batches,model=self.model
            )
            if "target" in dir(loss):
                loss.target = self.target

            _losses += [loss]

        return _losses

    @cached_property
    def schedulers(self):
        _schedulers = []

        for scheduler_config in self.trainer_config.scheduler_configs:
            _scheduler = SCHEDULER_REGISTRY[scheduler_config.scheduler_type](
                **dict(scheduler_config.specific_scheduler_config)
            )
            if "metrics" in dir(_scheduler):
                _scheduler.metrics = self.model
            if "target_action" in dir(_scheduler):
                _scheduler.target_action = self.target.dist.action

            _schedulers += [_scheduler]

        return _schedulers

    @property
    def sus_exact(self):
        return SusceptibilityExact(self.target.dist.action.beta, *self.dim).evaluate()

    @cached_property
    def observables_fn(self):
        return {
            obs: OBSERVABLE_REGISTRY[self.target_system][obs]()
            for obs in self.observables
        }

    @cached_property
    def ess_p_sampler(self):
        return get_ess_p_sampler(
            dim=self.dim,
            beta=self.target.dist.action.beta,
            batch_size=self.trainer_config.batch_size_eval,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        train_loaders = []
        for loss in self.losses:
            if hasattr(loss, "sampler"):
                train_loaders += [loss.sampler]
            else:
                raise ValueError("Unhandled sampler case")

        return train_loaders

    def training_step(self, batches, *args, **kwargs):

        losses_out_batch = []
        for loss, batch in zip(self.losses, batches):
            losses_out_batch += [loss.evaluate(batch)]

        x_samples = losses_out_batch[0]["x_samples"]
        losses = losses_out_batch[0]["loss_batch"]

        metrics_dict = {}
        for obs_name, obs_fn in self.observables_fn.items():
            obs_values = obs_fn.evaluate(x_samples)
            metrics_dict[obs_name] = obs_values.mean()
            self.logger.experiment.add_histogram(tag=obs_name, values=obs_values)

        metrics_dict.update({"loss": losses.mean(), "loss_std": losses.std()})
        self.metrics.add_batch_wise(metrics_dict)

        for scheduler in self.schedulers:
            scheduler.step()
            scheduler.log(self.logger.experiment)

        for key, value in metrics_dict.items():
            self.log(key, value)

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

    def on_train_epoch_end(self):

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

        self.log("lr", self.learning_rate)

        self.log(
            "loss_epoch",
            self.metrics.last_mean("loss", self.trainer_config.num_batches),
        )

        self.log(
            "loss_std_epoch",
            self.metrics.last_mean("loss_std", self.trainer_config.num_batches),
        )

        # if "von_mises" in self.config.flow_config.base_dist_config.type:
        #     with torch.no_grad():
        #         self.log(
        #             "base_dist/concentration",
        #             self.model.base_distribution.constraint_transform(
        #                 self.model.base_distribution.concentration
        #             ),
        #         )

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
