from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Literal, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from scipy import stats

from nfqr.data import ConditionConfig, MCMCConfig, MCMCPSampler, MCMCSamplerConfig
from nfqr.data.datasampler import FlowSampler
from nfqr.eval.evaluation import (
    estimate_ess_p_nip,
    estimate_obs_nip,
    estimate_obs_nmcmc,
)
from nfqr.normalizing_flows.flow import BareFlow, FlowConfig
from nfqr.normalizing_flows.loss.loss import elbo
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact,RotorTrajectorySamplerConfig
from nfqr.train.config import TrainerConfig
from nfqr.train.scheduler import BetaScheduler, BetaSchedulerConfig
from nfqr.mcmc.initial_config import InitialConfigConfig

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
        train_setup: Literal["reverse"] = "reverse",
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

        if train_setup == "reverse":
            self.training_step = self._training_step_reverse
            self.train_dataloader = partial(
                self._train_dataloader_reverse,
                batch_size=trainer_config.batch_size,
                num_batches=trainer_config.num_batches,
            )

        self.trainer_config = trainer_config
        self.dim = dim

        self.metrics = Metrics()

        self.schedulers = []
        for scheduler_config in trainer_config.scheduler_configs:
            if isinstance(scheduler_config, BetaSchedulerConfig):
                self.schedulers += [
                    BetaScheduler(
                        self.metrics, self.target.dist.action, **dict(scheduler_config)
                    )
                ]

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

        mcmc_sampler_config = MCMCSamplerConfig(
            mcmc_config=MCMCConfig(
                mcmc_alg="cluster",
                mcmc_type="wolff",
                observables="Chi_t",
                n_steps=1,
                dim=self.dim,
                action_config=ActionConfig(beta=self.target.dist.action.beta),
                n_burnin_steps=25000,
                n_traj_steps=3,
                out_dir=Path("./"),
                initial_config_sampler_config=InitialConfigConfig(trajectory_sampler_config=RotorTrajectorySamplerConfig(dim=self.dim,traj_type="hot"))
            ),
            condition_config=ConditionConfig(),
            batch_size=5000,
        )

        p_sampler = MCMCPSampler(
            sampler_configs=[mcmc_sampler_config],
            batch_size=self.trainer_config.batch_size_eval,
            elements_per_dataset=250000,
            subset_distribution=[1.0],
            num_workers=1,
            shuffle=True,
        )

        return p_sampler

    def _train_dataloader_reverse(self, batch_size, num_batches) -> TRAIN_DATALOADERS:

        train_loader = FlowSampler(
            batch_size=batch_size,
            num_batches=num_batches,
            model=self.model,
        )

        return train_loader

    def _training_step_reverse(self, batch, *args, **kwargs):

        x_samples, log_q_x = batch
        log_p = self.target.log_prob(x_samples)

        elbo_values = elbo(log_q=log_q_x, log_p=log_p)
        loss = elbo_values.mean()

        metrics_dict = {}
        for name, observable in self.observables_fn.items():
            metrics_dict[name] = observable.evaluate(x_samples).mean()

        metrics_dict.update({"loss": loss, "loss_std": elbo_values.std()})
        self.metrics.add_batch_wise(metrics_dict)

        for scheduler in self.schedulers:
            scheduler.step()

        self.log("beta", self.target.dist.action.beta)

        for key, value in metrics_dict.items():
            self.log(key, value)

        return {"loss": loss}

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
