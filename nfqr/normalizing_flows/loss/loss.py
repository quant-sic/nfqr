from torch.nn import Module


class ReverseKL(Module):
    """ """

    def __init__(self, model, target, batch_size):
        super().__init__()
        self.model = model
        self.target = target
        self.batch_size = batch_size

    def forward(self):

        samples, abs_log_det = self.model.sample_with_abs_log_det((self.batch_size,))

        loss = abs_log_det - self.target.log_prob(samples)

        return loss.mean()


def elbo(log_q, log_p):
    return log_q - log_p
