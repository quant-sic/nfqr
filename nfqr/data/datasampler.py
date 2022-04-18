from torch.utils.data import DataLoader

from nfqr.mcmc.base import MCMC


class FlowSampler(object):
    def __init__(self, batch_size, num_batches, model) -> None:
        self.num_batches = num_batches
        self.model = model
        self.batch_size = batch_size

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            yield self.model.sample_with_abs_log_det((self.batch_size,))

    def __len__(self):
        return self.num_batches


class MCMCSampler(object):
    def __init__(self, mcmc: MCMC, num_batches) -> None:
        self.num_batches = num_batches
        self.mcmc = mcmc
        self.mcmc.initialize()

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            yield from self.mcmc.run()

    def __len__(self):
        return self.num_batches


class ExtendedDLIterator(object):
    def __init__(self, data_loader: DataLoader) -> None:
        self.dl = data_loader
        self.iterator = iter(self.dl)

    def _new_iterator(self):
        self.iterator = iter(self.dl)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.iterator.next()
        except StopIteration:
            self._new_iterator()
            return self.iterator.next()

    def sample(self, device):
        sample = self.__next__()
        return sample.to(device)
