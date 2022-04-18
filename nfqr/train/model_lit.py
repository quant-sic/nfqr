import pytorch_lightning as pl
import torch

from nfqr.evaluation import estimate_ess_nip
from nfqr.mcmc.nmcmc import estimate_nmcmc_acc_rate
from nfqr.normalizing_flows.flow import U1CouplingChain, U1Flow
from nfqr.normalizing_flows.loss.loss import elbo
from nfqr.normalizing_flows.target_density import TargetDensity
from nfqr.target_systems.observable import ObservableRecorder
from nfqr.target_systems.rotor.rotor import TopologicalCharge, TopologicalSusceptibility


class LitFlow(pl.LightningModule):
    def __init__(
        self,
        size,
        coupling,
        split_type,
        net,
        num_layers,
        expressivity,
        coupling_specifiers,
        net_kwargs,
        base_distribution,
        target_action,
        train_setup,
        log_sample_shape,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_setup"])

        transform = U1CouplingChain(
            size=size,
            net=net,
            coupling=coupling,
            split_type=split_type,
            num_layers=num_layers,
            expressivity=expressivity,
            coupling_specifiers=coupling_specifiers,
            **net_kwargs
        )
        base = base_distribution(dim=size)
        self.model = U1Flow(base, transform)
        self.target = TargetDensity.boltzmann_from_action(target_action)

        self.observables = {
            "Q_mean": TopologicalCharge(),
            "Chi_t_mean": TopologicalSusceptibility(),
        }

        if train_setup == "reverse":
            self.training_step = self._training_step_reverse

    def _training_step_reverse(self, batch, *args, **kwargs):

        x_samples, log_q_x = batch
        log_p = self.target.log_prob(x_samples)

        elbo_values = elbo(log_q=log_q_x, log_p=log_p)
        loss = elbo_values.mean()

        for name, observable in self.observables.items():
            self.log(name, observable.evaluate(x_samples).mean())

        self.log("loss", loss)
        self.log("loss_std", elbo_values.std())

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_train_epoch_end(self):

        acc_rate = estimate_nmcmc_acc_rate(
            self.model, self.target, trove_size=5000, n_steps=100
        )
        self.log("acc_rate", acc_rate)

        ess_q = estimate_ess_nip(
            model=self.model, target=self.target, batch_size=5000, n_iter=10
        )
        self.log("ess/ess_q", ess_q)

    def training_epoch_end(self, train_outputs):

        self.log(
            "loss_epoch",
            sum(map(lambda output: output["loss"], train_outputs)) / len(train_outputs),
        )
