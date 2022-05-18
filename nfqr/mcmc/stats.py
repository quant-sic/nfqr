import numpy as np
import torch

from nfqr.mcmc.ac.ac import NoFluctuationsError, PrimaryAnalysis


def basic_integrated_ac(history):
    """
    Returns the autocorrelation for given history
    """
    N = len(history)
    history_c = history - history.mean()
    ac = np.correlate(history_c, history_c, mode="full")
    ac = ac[N - 1 :] / np.arange(N, 0, -1)
    normed_ac = ac / ac[0]

    integration_length = int(0.01 * N)
    tau_int = normed_ac[:integration_length].sum()

    return tau_int


def uw_analysis(history):

    if isinstance(history, torch.Tensor):
        history = history.numpy()

    try:
        history = history[None, :, None]
        analysis = PrimaryAnalysis(history, [np.prod(history.shape)], name="primary")
        analysis.mean()
        results = analysis.errors()

        mean = results.value.item()
        error = results.dvalue[0]
        tau = results.tau_int[0]
        dtau = results.dtau_int[0]

    except NoFluctuationsError:
        mean, error, tau, dtau = 0, 0, 0, 0

    return mean, error, tau, dtau


def get_mcmc_statistics(history):

    mean, error_unew, tau_unew, dtau_unew = uw_analysis(history)

    return {
        "mean": mean,
        "error": error_unew,
        "tau_int": tau_unew,
        "dtau_int": dtau_unew,
    }
