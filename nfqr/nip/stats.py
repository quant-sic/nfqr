import math

import numpy as np
import torch

from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def remove_nans_and_infs(weights, history=None, nan_threshold=0.01):

    nan_mask = torch.isinf(weights) | torch.isnan(weights)
    if history is not None:
        nan_mask = nan_mask | torch.isnan(history) | torch.isinf(weights)
        cleaned_history = history[~nan_mask]
    else:
        cleaned_history = None

    cleaned_weights = weights[~nan_mask]

    if len(cleaned_weights) < len(weights):
        logger.info(
            f"{len(weights)-len(cleaned_weights)} nans have been removed from weights"
        )
        if (1 - len(cleaned_weights) / len(weights)) > nan_threshold:
            raise RuntimeError("Too many nans in weights")

    return cleaned_weights, cleaned_history


def get_impsamp_statistics(history, unnormalized_log_weights):
    with torch.no_grad():

        cleaned_log_weights, history = remove_nans_and_infs(
            unnormalized_log_weights, history
        )
        cleaned_weights = calc_imp_weights(cleaned_log_weights)
        weights = cleaned_weights / cleaned_weights.mean()

        mean = (weights * history).mean()
        ess = calc_ess_q(weights=weights)

        error = (weights * history).std() / math.sqrt(len(history))

        sq_mean = (weights * history**2).mean()
        std = torch.sqrt(abs(sq_mean - mean**2) / len(history))
        error_ess = std / math.sqrt(ess)

        return {
            "mean": mean.item(),
            "error": error.item(),
            "error_ess": error_ess.item(),
            "ess_q": ess.item(),
        }


def calc_imp_weights(log_weight_hist, subtract_max=True):

    if subtract_max:
        # use exp-norm trick:
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
        log_weight_hist_max, _ = log_weight_hist.max(dim=0)
        log_weight_hist = log_weight_hist - log_weight_hist_max

    weight_hist = log_weight_hist.exp()

    return weight_hist


def calc_ess_q(weights):

    ess_q = 1 / (weights**2).mean()

    return ess_q


def calc_ess_q_from_unnormalized_log_weights(unnormalized_log_weights):

    cleaned_log_weights = remove_nans_and_infs(unnormalized_log_weights)
    cleaned_weights = calc_imp_weights(cleaned_log_weights)
    weights = cleaned_weights / cleaned_weights.mean()

    return calc_ess_q(weights)
