import math

import numpy as np
import torch

from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


def remove_nans_and_infs(weights, history=None,log_p=None, nan_threshold=0.01):

    nan_mask = torch.isinf(weights) | torch.isnan(weights)
    
    for additional in (history,log_p):
        if additional is not None:
            nan_mask = nan_mask | torch.isnan(additional) | torch.isinf(additional)
    
    if history is not None:
        cleaned_history = history[~nan_mask]
    else:
        cleaned_history = None

    if log_p is not None:
        cleaned_log_p = log_p[~nan_mask]
    else:
        cleaned_log_p = None

    cleaned_weights = weights[~nan_mask]

    if len(cleaned_weights) < len(weights):
        logger.info(
            f"{len(weights)-len(cleaned_weights)} nans have been removed from weights"
        )
        if (1 - len(cleaned_weights) / len(weights)) > nan_threshold:
            raise RuntimeError("Too many nans in weights")

    return_values = list(filter(lambda v: v is not None,(cleaned_weights, cleaned_history,cleaned_log_p)))
    if len(return_values)==1:
        return_values = return_values[0]

    return return_values


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

    cleaned_log_weights= remove_nans_and_infs(unnormalized_log_weights)
    cleaned_weights = calc_imp_weights(cleaned_log_weights)
    weights = cleaned_weights / cleaned_weights.mean()

    return calc_ess_q(weights)


def calc_ess_p_from_unnormalized_log_weights(
    unnormalized_log_weights, cut_quantiles=[0, 1]
):

    cleaned_log_weights= remove_nans_and_infs(unnormalized_log_weights)
    cleaned_log_weights = cleaned_log_weights.to(torch.float64)

    quantile_mask = (
        cleaned_log_weights >= torch.quantile(cleaned_log_weights, q=cut_quantiles[0])
    ) & (cleaned_log_weights <= torch.quantile(cleaned_log_weights, q=cut_quantiles[1]))

    filtered_weights = cleaned_log_weights[quantile_mask]

    w_shifted = filtered_weights - filtered_weights.max()
    w_inv_shifted = -filtered_weights - (-filtered_weights).max()

    log_weights_sum = torch.logsumexp(w_shifted, dim=-1)
    log_weights_sum_inv = torch.logsumexp(w_inv_shifted, dim=-1)

    log_scale_factor = (
        2 * np.log(len(filtered_weights))
        - filtered_weights.max()
        - (-filtered_weights).max()
    )

    log_ess_p = log_scale_factor - log_weights_sum - log_weights_sum_inv
    ess_p = log_ess_p.exp()

    return {"log_ess_p": log_ess_p.item(), "ess_p": ess_p.item()}

def calc_free_energy(unnormalized_log_weights,dim):
    
    cleaned_log_weights = remove_nans_and_infs(unnormalized_log_weights)
    cleaned_log_weights = cleaned_log_weights.to(torch.float64)

    return (-(cleaned_log_weights.logsumexp(dim=0) - math.log(len(cleaned_log_weights)))/np.prod(dim)).item()

def calc_std_free_energy( unnormalized_log_weights,dim):

    cleaned_log_weights = remove_nans_and_infs(unnormalized_log_weights)
    cleaned_weights = calc_imp_weights(cleaned_log_weights)
    weights = cleaned_weights / cleaned_weights.mean()

    n = len(weights)
    var = 1 / n * ((weights**2).mean() - 1) / np.prod(dim)
    
    return torch.sqrt(var).item()

def calc_entropy(unnormalized_log_weights, log_p,dim):
    
    cleaned_log_weights,cleaned_log_p = remove_nans_and_infs(unnormalized_log_weights,log_p)
    log_q_x = cleaned_log_p - cleaned_log_weights
    cleaned_weights = calc_imp_weights(cleaned_log_weights)
    weights = cleaned_weights / cleaned_weights.mean()
    
    n = len(cleaned_log_weights)
    s = cleaned_log_weights - torch.logsumexp(cleaned_log_weights, dim=0) + log_q_x

    return (-( (weights * s).sum() + math.log(n)) / (np.prod(dim))).item()

def calc_std_entropy(unnormalized_log_weights, log_p,dim):
    """
    -1 + Eg2w2 - 2 Egw2 (-1 + g) + Ew2 (1 - 2 g + g^2) where g = beta E and w are normalized impweights
    """
    cleaned_log_weights, cleaned_log_p= remove_nans_and_infs(unnormalized_log_weights,log_p)
    cleaned_weights = calc_imp_weights(cleaned_log_weights)
    weights = cleaned_weights / cleaned_weights.mean()

    n = len(weights)
    g = (-cleaned_log_p * weights).sum()
    w = n * weights  # normalized weights w/o any additional n factor

    Eg2w2 = ((w * cleaned_log_p)**2).mean()
    Egw2 = (-cleaned_log_p * w**2).mean()
    Ew2 = (w**2).mean()

    var = (-1 + Eg2w2 - 2 * Egw2 * (-1 + g) + Ew2 * (1 - 2 * g + g**2)) / n

    return (torch.sqrt(var) / (np.prod(dim))).item()
