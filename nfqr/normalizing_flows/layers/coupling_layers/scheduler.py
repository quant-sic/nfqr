import numpy as np
import torch
from pydantic import BaseModel

from nfqr.normalizing_flows.layers.coupling_layers.couplings import ResidualCoupling


class ResidualCouplingScheduler(object):
    def __init__(self, n_steps_initial, reduction_factor, patience) -> None:

        self.n_steps_initial = n_steps_initial
        self.reduction_factor = reduction_factor
        self.patience = patience

        self.current_step = 0
        self.n_steps_since_action = 0

    @property
    def log_stats(self):
        return {}

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise RuntimeError("model not set yet")
        else:
            return self._model

    @model.setter
    def model(self, m):
        self._model = m

    def step(self):

        self.current_step += 1

        if self.current_step < self.n_steps_initial:
            return

        self.n_steps_since_action += 1

        if self.n_steps_since_action < self.patience:
            return

        for module in self.model.modules():
            if not isinstance(module, ResidualCoupling):
                continue

            elif hasattr(module, "rho_unnormalized"):
                # avoid running into nans
                if (
                    torch.softmax(module.rho_unnormalized, dim=-1)[
                        module.rho_assignment["id"]
                    ]
                    < 1e-2
                ):
                    module.rho_unnormalized[module.rho_assignment["id"]] = -torch.inf

                else:
                    # calculated change to unnormalized rho. for softmax simplex transform
                    x = 0.5 * (
                        torch.log(
                            self.reduction_factor
                            * module.rho_unnormalized.exp()[module.rho_assignment["id"]]
                            / (
                                module.rho_unnormalized.exp()[
                                    module.rho_assignment["diff"]
                                ]
                                + (1 - self.reduction_factor)
                                * module.rho_unnormalized.exp()[
                                    module.rho_assignment["id"]
                                ]
                            )
                        )
                        + module.rho_unnormalized[module.rho_assignment["diff"]]
                        - module.rho_unnormalized[module.rho_assignment["id"]]
                    )

                    module.rho_unnormalized[module.rho_assignment["id"]] += x
                    module.rho_unnormalized[module.rho_assignment["diff"]] -= x

            else:
                pass


class ResidualCouplingSchedulerConfig(BaseModel):

    n_steps_initial: int
    reduction_factor: float
    patience: int
