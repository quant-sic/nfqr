from typing import Dict, Union

import numpy as np
from pydantic import BaseModel

from nfqr.registry import StrRegistry
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)

SCHEDULER_REGISTRY = StrRegistry("scheduler")


@SCHEDULER_REGISTRY.register("beta")
class BetaScheduler(object):
    def __init__(
        self,
        metrics,
        target_action,
        target_beta,
        damping_constant: Dict[str, Union[int, float]],
        max_observed_change_rate: Dict[str, Union[int, float]],
        change_rate: float,
        cooldown_steps: int,
        metric_window_length: int,
    ) -> None:

        self.target_action = target_action
        self.metrics = metrics

        self.metric_window_length = metric_window_length
        self.target_beta = target_beta
        self.change_rate = change_rate
        self.damping_constant = damping_constant
        self.max_observed_change_rate = max_observed_change_rate

        self.n_steps_since_change = 0
        self.cooldown_steps = cooldown_steps

    def step(self):

        self.n_steps_since_change += 1
        if not (self.n_steps_since_change > self.cooldown_steps):
            return
        else:
            # check whether sufficiently close to target to avoid overshooting loop
            if abs(self.target_beta - self.target_action.beta) < self.change_rate:
                self.target_action.beta = self.target_beta
                return

            # propose change of beta
            beta_change_proposed = self.change_rate * np.sign(
                self.target_beta - self.target_action.beta
            )

            # introduce additional damping or cancel change if required conditions are not met
            damping_exponent = 0
            for key, damping_c in self.damping_constant.items():
                logger.info(
                    f"slope {self.metrics.last_slope(key, self.metric_window_length)}"
                )
                slope = self.metrics.last_slope(key, self.metric_window_length)
                logger.info(self.max_observed_change_rate[key])
                logger.info(abs(slope) < self.max_observed_change_rate[key])
                slope = (
                    slope if abs(slope) < self.max_observed_change_rate[key] else np.inf
                )
                damping_exponent += abs(damping_c * slope)

            logger.info(damping_exponent)
            self.target_action.beta += beta_change_proposed * np.exp(-damping_exponent)

            self.n_steps_since_change = 0


class BetaSchedulerConfig(BaseModel):

    target_beta: float
    damping_constant: Dict[str, float]
    max_observed_change_rate: Dict[str, float]
    change_rate: float
    cooldown_steps: int
    metric_window_length: int
