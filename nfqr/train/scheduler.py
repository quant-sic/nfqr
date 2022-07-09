import inspect
from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Union

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
        target_beta,
        damping_constant: Dict[str, Union[int, float]],
        max_observed_change_rate: Dict[str, Union[int, float]],
        change_rate: float,
        cooldown_steps: int,
        metric_window_length: int,
    ) -> None:

        self.metric_window_length = metric_window_length
        self.target_beta = target_beta
        self.change_rate = change_rate
        self.damping_constant = damping_constant
        self.max_observed_change_rate = max_observed_change_rate

        self.n_steps_since_change = 0
        self.cooldown_steps = cooldown_steps

    @property
    def metrics(self):
        if not hasattr(self, "_metrics"):
            raise RuntimeError("metrics not set yet")
        else:
            return self._metrics

    @metrics.setter
    def metrics(self, m):
        self._metrics = m

    @property
    def target_action(self):
        if not hasattr(self, "_target_action"):
            raise RuntimeError("target_action not set yet")
        else:
            return self._target_action

    @target_action.setter
    def target_action(self, ta):
        self._target_action = ta

    @property
    def log_stats(self):
        return {"beta": self.target_action.beta}

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
                logger.debug(
                    f"slope {self.metrics.last_slope(key, self.metric_window_length)}"
                )
                slope = self.metrics.last_slope(key, self.metric_window_length)
                logger.debug(self.max_observed_change_rate[key])
                logger.debug(abs(slope) < self.max_observed_change_rate[key])
                slope = (
                    slope if abs(slope) < self.max_observed_change_rate[key] else np.inf
                )
                damping_exponent += abs(damping_c * slope)

            logger.debug(damping_exponent)
            self.target_action.beta += beta_change_proposed * np.exp(-damping_exponent)

            self.n_steps_since_change = 0


@SCHEDULER_REGISTRY.register("loss")
class LossScheduler(object):
    def __init__(
        self,
        cooldown_steps: int,
        initial_alphas,
        target_alphas,
        alpha_step_method,
        n_schedule_steps,
        change_rate,
        metric_window_length,
        max_observed_change_rate,
        damping_constant,
    ):

        self.n_steps_since_change = 0
        self.cooldown_steps = cooldown_steps
        self.current_step = 0

        self.alphas = np.array(initial_alphas)
        self.target_alphas = np.array(target_alphas)
        self.n_schedule_steps = n_schedule_steps

        # self.alpha_step_method = self.comb_methods[alpha_step_method]
        self.alpha_step_method = {"max_change_comb": self.max_change_comb}[
            alpha_step_method
        ]

        self.change_rate = change_rate
        self.metric_window_length = metric_window_length
        self.max_observed_change_rate = max_observed_change_rate
        self.damping_constant = damping_constant

    @property
    def metrics(self):
        if not hasattr(self, "_metrics"):
            raise RuntimeError("metrics not set yet")
        else:
            return self._metrics

    @metrics.setter
    def metrics(self, m):
        self._metrics = m

    @property
    def log_stats(self):
        return {f"alpha/{idx}": alpha for idx, alpha in enumerate(self.alphas)}

    @cached_property
    def sufficient_target_distance(self):
        if self.alpha_step_method == self.max_change_comb:
            return self.change_rate
        else:
            return 5 / self.n_schedule_steps

    def step(self):

        self.current_step += 1
        self.n_steps_since_change += 1
        if not (self.n_steps_since_change > self.cooldown_steps):
            return
        else:
            # check whether sufficiently close to target
            if (
                abs(self.target_alphas - self.alphas) < self.sufficient_target_distance
            ).all():
                self.alphas = self.target_alphas
                return

            self.alpha_step_method()
            self.n_steps_since_change = 0

    def evaluate(self, losses_out):

        if len(self.alphas) != len(losses_out):
            raise ValueError(
                f"Number of alpahs {len(self.alphas)} does not match number of calculated losses {len(losses_out)}"
            )

        metrics = defaultdict(list)
        for loss_out in losses_out:
            for k, m in loss_out.items():
                metrics[k] += [m]

        if not len(set(list(map(len, metrics.values())))) == 1:
            raise ValueError("Currently all losses must yield the same keys")

        combined_metrics = {
            k: sum(self.alphas[i] * m_list[i] for i in range(len(self.alphas)))
            for k, m_list in metrics.items()
        }

        return combined_metrics

    def linear_comb(self):
        if self.n_schedule_steps == self.current_step:
            self.alphas = self.target_alphas
        else:
            self.alphas += (self.target_alphas - self.alphas) / max(
                self.n_schedule_steps - self.current_step, 1
            )

    # def tanh_comb(step, interval_length, alpha_start, alpha_end, tanh_end):
    #     return (
    #         np.tanh(
    #             (step - interval_length / 2)
    #             * 2
    #             * np.arctanh(tanh_end)
    #             / interval_length
    #         )
    #         / tanh_end
    #         * 0.5
    #         + 0.5
    #     ) * (alpha_end - alpha_start) + alpha_start

    def max_change_comb(self):

        # propose change of beta
        alphas_change_proposed = self.change_rate * np.sign(
            self.target_alphas - self.alphas
        )

        # introduce additional damping or cancel change if required conditions are not met
        damping_exponent = 0
        for key, damping_c in self.damping_constant.items():
            logger.debug(
                f"slope {self.metrics.last_slope(key, self.metric_window_length)}"
            )
            slope = self.metrics.last_slope(key, self.metric_window_length)
            logger.debug(self.max_observed_change_rate[key])
            logger.debug(abs(slope) < self.max_observed_change_rate[key])
            slope = slope if abs(slope) < self.max_observed_change_rate[key] else np.inf
            damping_exponent += abs(damping_c * slope)

        logger.debug(damping_exponent)
        self.alphas += alphas_change_proposed * np.exp(-damping_exponent)

    # @staticmethod
    # def half_tanh_comb(step, interval_length, alpha_start, alpha_end, tanh_end):
    #     return (np.tanh((step) * np.arctanh(tanh_end) / interval_length) / tanh_end) * (
    #         alpha_end - alpha_start
    #     ) + alpha_start

    def constant_comb(self):
        pass

    @property
    def comb_methods(self):
        return dict(
            filter(
                lambda func: "comb" in func[0],
                inspect.getmembers(LossScheduler, predicate=inspect.isfunction),
            )
        )


@SCHEDULER_REGISTRY.register("default_loss")
class LossScheduler(object):
    @property
    def log_stats(self):
        return {}

    def step(self):
        pass

    def evaluate(self, losses_out):
        return losses_out[0]


class LossSchedulerConfig(BaseModel):

    cooldown_steps: int
    initial_alphas: List[float]
    target_alphas: List[float]
    alpha_step_method: str
    n_schedule_steps: int
    change_rate: float
    metric_window_length: int
    max_observed_change_rate: Dict[str, float]
    damping_constant: Dict[str, float]


class BetaSchedulerConfig(BaseModel):

    target_beta: float
    damping_constant: Dict[str, float]
    max_observed_change_rate: Dict[str, float]
    change_rate: float
    cooldown_steps: int
    metric_window_length: int


class SchedulerConfig(BaseModel):

    scheduler_type: SCHEDULER_REGISTRY.enum
    specific_scheduler_config: Union[BetaSchedulerConfig, LossSchedulerConfig]
