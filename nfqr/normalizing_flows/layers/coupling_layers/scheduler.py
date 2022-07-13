import numpy as np
import torch
from pydantic import BaseModel


class ResidualCouplingScheduler(object):
    def __init__(self, n_steps_initial, reduction_factor, patience) -> None:

        self.n_steps_initial = n_steps_initial
        self.reduction_factor = reduction_factor
        self.patience = patience

        self.current_step = 0
        self.n_steps_since_action = 0

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise RuntimeError("model not set yet")
        else:
            return self._model

    @model.setter
    def metrics(self, m):
        self._model = m

    def step(self):

        self.current_step += 1

        if self.current_step < self.n_steps_initial:
            return

        self.n_steps_since_action += 1

        if self.n_steps_since_action < self.patience:
            return

        for module in self.model.modules():
            if hasattr(module, "rho_unnormalized"):

                # calculated change to unnormalized rho. for softmax simplex transform
                x = 0.5 * (
                    np.log(
                        self.reduction_factor
                        * module.rho_unnormalized.exp()[1]
                        / (
                            module.rho_unnormalized.exp()[0]
                            + (1 - self.reduction_factor)
                            * module.rho_unnormalized.exp()[1]
                        )
                    )
                    + module.rho_unnormalized[0]
                    - module.rho_unnormalized[1]
                )

                module.rho_unnormalized += torch.tensor(
                    [x, -x], device=module.rho_unnormalized.device
                )


class ResidualCouplingSchedulerConfig(BaseModel):

    n_steps_initial: int
    reduction_factor: float
    patience: int
