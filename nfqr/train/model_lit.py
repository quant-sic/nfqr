from functools import cached_property, partial
from typing import List, Literal, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

from nfqr.data.datasampler import FlowSampler
from nfqr.eval.evaluation import estimate_obs_nip, estimate_obs_nmcmc
from nfqr.normalizing_flows.flow import BareFlow, FlowConfig
from nfqr.normalizing_flows.loss.loss import elbo
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems import ACTION_REGISTRY, OBSERVABLE_REGISTRY, ActionConfig
from nfqr.target_systems.rotor import SusceptibilityExact
from nfqr.train.config import TrainerConfig


class LitFlow(pl.LightningModule):
    def __init__(
        self,
        dim: Tuple[int],
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

        self.sus_exact = SusceptibilityExact(action_config.beta, *dim).evaluate()

        if train_setup == "reverse":
            self.training_step = self._training_step_reverse
            self.train_dataloader = partial(
                self._train_dataloader_reverse,
                batch_size=trainer_config.batch_size,
                num_batches=trainer_config.num_batches,
            )

    @cached_property
    def observables_fn(self):
        return {
            obs: OBSERVABLE_REGISTRY[self.target_system][obs]()
            for obs in self.observables
        }

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

        for name, observable in self.observables_fn.items():
            self.log(name, observable.evaluate(x_samples).mean())

        self.log("loss", loss)
        loss_std = elbo_values.std()
        self.log("loss_std", loss_std)

        return {"loss": loss, "loss_std": loss_std}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):

        stats_nmcmc = self.estimate_obs_nmcmc(
            batch_size=5000,
            n_iter=1,
        )

        stats_nip = self.estimate_obs_nip(
            batch_size=5000,
            n_iter=10,
        )

        for sampler, stats in zip(("nip", "nmcmc"), (stats_nip, stats_nmcmc)):
            for key, values in stats.items():
                if not isinstance(values, dict):
                    values = {"valid": values}
                for stat, value in values.items():
                    self.log(f"{sampler}/{key}/{stat}", value)
                    if "Chi_t" in key and stat == "mean":
                        self.log(
                            f"{sampler}/{key}/abs_diff_to_exact",
                            abs(value - self.sus_exact),
                        )

        self.log("lr", self.learning_rate)

        # if "von_mises" in self.config.flow_config.base_dist_config.type:
        #     with torch.no_grad():
        #         self.log(
        #             "base_dist/concentration",
        #             self.model.base_distribution.constraint_transform(
        #                 self.model.base_distribution.concentration
        #             ),
        #         )

    def training_epoch_end(self, train_outputs):

        self.log(
            "loss_epoch",
            sum(map(lambda output: output["loss"], train_outputs)) / len(train_outputs),
        )

        # add beta scheduling

    def estimate_obs_nip(self, batch_size, n_iter):

        stats_nip = estimate_obs_nip(
            model=self.model,
            target=self.target,
            observables=self.observables,
            batch_size=batch_size,
            n_iter=n_iter,
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
