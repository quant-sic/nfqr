import numpy as np
import torch
import unew.analysis as un
from unew.errors import NoFluctuationsError
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)

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

    def run(self):
        for _ in self:
            pass

    def get_stats(self):
        return {
            "acceptance_rate": self.acceptance_rate,
            "n_steps": self.n_current_steps,
            "obs_stats": self.observable_rec.aggregate(),
        }


def basic_integrated_ac(history):
    """
    Returns the autocorrelation for given history
    """
    N = len(history)
    history_c = history - history.mean()
    ac = np.correlate(history_c, history_c, mode="full")
    ac = ac[N - 1 :] / np.arange(N, 0, -1)
    normed_ac = ac / ac[0]

    integration_length = int(.1 * N)
    tau_int = normed_ac[:integration_length].sum()

    return tau_int


def uw_analysis(history,max_rep_size=1e6):

    # floored
    # rep_size = int(min(max_rep_size,len(history)))

    # # max not necessary, but makes it manifest
    # num_rep = max(int(len(history)/rep_size),1)

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

    except NoFluctuationsError:
        mean, error, tau, dtau = 0, 0, 0, 0

    return mean, error, tau, dtau


def get_mcmc_statistics(history):

    mean, error_unew, tau_unew, dtau_unew = uw_analysis(history)
    tau_int_basic = basic_integrated_ac(history)

    return {"mean": mean, "error": error_unew, "tau_int": tau_unew, "dtau_int": dtau_unew,"tau_int_basic":tau_int_basic}
