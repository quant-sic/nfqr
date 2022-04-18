import torch

from nfqr.utils.misc import create_logger

# from tqdm.autonotebook import tqdm


logger = create_logger(__name__)


def calc_imp_weights(
    log_weight_hist, subtract_max=True, include_factor_N=False, divide_by_mean=False
):
    """
    Calculates normalized importance weights from logarithm of unnormalized weights.
    If include_factor_N is set, then the result will be multiplied by the number of weights, s.t. the expectation is
    simply the sum of weights and history. Otherwise, the average instead of the sum has to be taken.
    """

    if subtract_max:
        # use exp-norm trick:
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
        log_weight_hist_max, _ = log_weight_hist.max(dim=0)
        log_weight_hist = log_weight_hist - log_weight_hist_max

    weight_hist = log_weight_hist.exp()

    if include_factor_N:
        weight_hist = weight_hist / weight_hist.sum(dim=0)
    elif divide_by_mean:
        weight_hist = weight_hist / weight_hist.mean(dim=0)

    return weight_hist


class NeuralImportanceSampler:
    def __init__(self, model, target, n_iter, batch_size=2000):
        self.batch_size = batch_size
        self.model = model
        self.target = target
        self.n_iter = n_iter

        # set model to evaluation mode
        self.model.eval()

    def __iter__(self):

        for _ in range(self.n_iter):
            yield self.step()

    @torch.no_grad()
    def step(self):

        x_samples, log_q_x = self.model.sample_with_abs_log_det((self.batch_size,))
        log_p = self.target.log_prob(x_samples)

        log_weights = log_p - log_q_x

        return x_samples, log_weights
