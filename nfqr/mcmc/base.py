import numpy as np
import torch
import unew.analysis as un


class MCMC(object):
    def __init__(self, n_steps) -> None:
        self.n_steps = n_steps
        self._config = None
        self.n_accepted = 0
        self.n_current_steps = 0

    def step(self):
        pass

    def initialize(self):
        pass

    @property
    def current_config(self):
        return self._config

    @current_config.setter
    def current_config(self, new_current_config):
        self._config = new_current_config

    def __iter__(self):

        self.initialize()

        for _ in range(self.n_steps):
            self.step()
            self.n_current_steps += 1
            yield self.current_config

    def run_entire_chain(self):
        for _ in self:
            pass

    @property
    def acceptance_ratio(self):
        return self.n_accepted / self.n_current_steps


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


def get_mcmc_statistics(history):
    mean, error, tau, dtau = uw_analysis(history)
    return {"mean": mean, "error": error, "tau_int": tau, "dtau_int": dtau}
