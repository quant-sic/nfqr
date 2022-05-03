import math

import numpy as np
import torch
import unew.analysis as un

from nfqr.nip.nip import calc_ess_q


def basic_integrated_ac(history):
    """
    Returns the autocorrelation for given history
    """
    N = len(history)
    history_c = history - history.mean()
    ac = np.correlate(history_c, history_c, mode="full")
    ac = ac[N - 1 :] / np.arange(N, 0, -1)
    return ac / ac[0]


def uw_analysis(history):

    if isinstance(history, torch.Tensor):
        history = history.numpy()

    try:
        history = history[None, :, None]
        analysis = un.PrimaryAnalysis(history, [np.prod(history.shape)], name="primary")
        analysis.mean()
        results = analysis.errors()

        mean = results.value.item()
        error = results.dvalue[0]
        tau = results.tau_int[0]
        dtau = results.dtau_int[0]

    except un.errors.NoFluctuationsError:
        mean, error, tau, dtau = 0, 0, 0, 0

    return mean, error, tau, dtau


def get_iid_statistics(history):

    with torch.no_grad():
        count = history.shape[0]
        mean = history.mean()
        sq_mean = (history**2).mean()

        std = torch.sqrt(abs(sq_mean - mean**2))
        err = std / math.sqrt(count)

        return {"mean": mean.item(), "error": err.item()}


def get_mcmc_statistics(history):
    mean, error, tau, dtau = uw_analysis(history)
    return {"mean": mean, "error": error, "tau_int": tau, "dtau_int": dtau}


def get_impsamp_statistics(history, unnormalized_weights):
    with torch.no_grad():

        weights = unnormalized_weights / unnormalized_weights.mean()

        mean = (weights * history).mean()

        sq_mean = (weights * history**2).mean()
        std = torch.sqrt(abs(sq_mean - mean**2))

        ess = calc_ess_q(unnormalized_weights=unnormalized_weights)
        err = std / math.sqrt(ess)

        return {"mean": mean.item(), "error": err.item(), "ess_q": ess.item()}
