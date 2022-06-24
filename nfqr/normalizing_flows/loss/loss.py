from typing import Literal, Union

from pydantic import BaseModel

from nfqr.data import PSampler, PSamplerConfig
from nfqr.registry import StrRegistry

LOSS_REGISTRY = StrRegistry("loss")


@LOSS_REGISTRY.register("reverse")
class ReverseKL(object):
    def __init__(self, **kwargs) -> None:
        pass

    @staticmethod
    def elbo(log_q, log_p):
        return log_q - log_p

    @property
    def target(self):
        if not hasattr(self, "_target"):
            raise RuntimeError("Target not set yet")
        else:
            return self._target

    @target.setter
    def target(self, t):
        self._target = t

    def evaluate(self, batch):

        x_samples, log_q_x = batch
        log_p = self.target.log_prob(x_samples)
        elbo_values = self.elbo(log_q=log_q_x, log_p=log_p)

        return {"loss_batch": elbo_values, "x_samples": x_samples}


@LOSS_REGISTRY.register("forward")
class ForwardKL(object):
    def __init__(self, p_sampler_config, **kwargs) -> None:
        self._p_sampler = PSampler(**dict(p_sampler_config))

    @property
    def p_sampler(self):
        return self._p_sampler

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not set yet")
        else:
            return self._model

    @model.setter
    def model(self, m):
        self._model = m

    def evaluate(self, batch):

        x_samples = batch
        nlog_q_x = -self.model.log_prob(x_samples.clone().detach())

        return {"loss_batch": nlog_q_x, "x_samples": x_samples}


class ReverseKLConfig(BaseModel):
    _name = "reverse_kl_config"


class ForwardKLConfig(BaseModel):

    _name = "forward_kl_config"
    p_sampler_config: PSamplerConfig


class LossConfig(BaseModel):

    loss_type: LOSS_REGISTRY.enum
    specific_loss_config: Union[ReverseKLConfig, ForwardKLConfig]
