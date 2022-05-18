import torch
from tqdm.autonotebook import tqdm

from nfqr.nip.stats import calc_imp_weights, get_impsamp_statistics
from nfqr.sampler import Sampler
from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class NeuralImportanceSampler(Sampler):
    def __init__(
        self,
        model,
        observables,
        target,
        n_iter,
        out_dir,
        target_system="qr",
        batch_size=2000,
    ):
        super(NeuralImportanceSampler, self).__init__(
            observables=observables, target_system=target_system, out_dir=out_dir
        )

        self.batch_size = batch_size
        self.model = model
        self.target = target
        self.n_iter = n_iter

        # set model to evaluation mode
        self.model.eval()

    def run(self):

        for _ in tqdm(range(self.n_iter), desc="Running NIP"):
            self.step()

    @torch.no_grad()
    def step(self):

        x_samples, log_q_x = self.model.sample_with_abs_log_det((self.batch_size,))
        log_p = self.target.log_prob(x_samples)

        log_weights = log_p - log_q_x

        self.observables_rec.record_config_with_log_weight(x_samples, log_weights)

    @property
    def unnormalized_log_weights(self):
        self.observables_rec["log_weights"]

    def _evaluate_obs(self, obs):

        observable_data = self.observables_rec[obs]
        prepared_observable_data = self.observables_rec.observables[obs].prepare(
            observable_data
        )

        config_log_weights_unnormalized = self.observables_rec["log_weights"]

        assert len(config_log_weights_unnormalized) == len(observable_data)

        stats = get_impsamp_statistics(
            prepared_observable_data, config_log_weights_unnormalized
        )

        stats_postprocessed = self.observables_rec.observables[obs].postprocess(stats)

        # else:
        #     stats = self.stats_function(prepared_observable_data)

        return stats_postprocessed
