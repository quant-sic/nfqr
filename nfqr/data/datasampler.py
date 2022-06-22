from asyncio.log import logger
from tkinter import N
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from nfqr.data.condition import SampleCondition
from nfqr.data.dataset import SamplesDataset, lmdb_worker_init_fn
from nfqr.mcmc import MCMC_REGISTRY
from nfqr.mcmc.config import MCMCConfig

from .config import ConditionConfig, MCMCSamplerConfig


class MultipleDatasetsBatchSampler(BatchSampler):
    def __init__(
        self,
        common_dataset,
        distribution,
        num_batches: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        assert len(distribution) == len(common_dataset.datasets)

        self.common_dataset = common_dataset
        self.distribution = distribution
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):

        if self.shuffle:

            for _ in range(self.num_batches):
                datasets = np.random.multinomial(
                    self.batch_size, self.distribution, size=1
                ).flatten()

                idx_tuple_list = []
                for dataset_idx, (num_samples, _set) in enumerate(
                    zip(datasets, self.common_dataset.datasets)
                ):
                    samples_idx = np.random.randint(0, len(_set), size=num_samples)

                    idx_tuple_list += [
                        (int(dataset_idx), int(_samples_idx))
                        for _samples_idx in samples_idx
                    ]

                yield idx_tuple_list

        else:
            current_idx_in_dataset = np.zeros(len(self.common_dataset.datasets))
            dataset_lengths = np.array(
                [len(_set) for _set in self.common_dataset.datasets]
            )

            for _ in range(self.num_batches):

                idx_tuple_list = []

                datasets = np.random.multinomial(
                    self.batch_size, self.distribution, size=1
                ).flatten()

                next_idx = current_idx_in_dataset + datasets

                for dataset_idx, _set in enumerate(self.common_dataset.datasets):

                    samples_idx = (
                        np.arange(
                            current_idx_in_dataset[dataset_idx], next_idx[dataset_idx]
                        )
                        % dataset_lengths[dataset_idx]
                    )

                    idx_tuple_list += [
                        (int(dataset_idx), int(_samples_idx))
                        for _samples_idx in samples_idx
                    ]

                next_idx = np.mod(next_idx, dataset_lengths)
                current_idx_in_dataset = next_idx

                yield idx_tuple_list

    def __len__(self):
        return self.num_batches


class FlowSampler(object):
    def __init__(self, batch_size, num_batches, model) -> None:
        self.num_batches = num_batches
        self.model = model
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.model.sample_with_abs_log_det((self.batch_size,))

    def __len__(self):
        return self.num_batches


class MCMCSampler(object):
    def __init__(
        self,
        mcmc_config: MCMCConfig,
        condition_config: ConditionConfig,
        batch_size: int,
        num_batches: int = 1,
    ) -> None:
        self.num_batches = num_batches
        self.mcmc = MCMC_REGISTRY[mcmc_config.mcmc_alg][mcmc_config.mcmc_type](
            **dict(mcmc_config)
        )
        self.condition = SampleCondition(**dict(condition_config))
        self.batch_size = batch_size
        self.mcmc.initialize()

    @property
    def sampler_specs(self):
        return {**self.mcmc.data_specs, "condition": str(self.condition)}

    def sample_batch(self, max_batch_repetitions=10, n_times_reinit=3):

        batch = []

        for _ in range(n_times_reinit):
            for _ in range(self.batch_size * max_batch_repetitions):

                self.mcmc.step(record_observables=False)

                if self.condition.evaluate(self.mcmc.current_config):
                    batch += [self.mcmc.current_config.detach().clone()]
                    if len(batch) >= self.batch_size:
                        return torch.concat(batch, dim=0)

            logger.info("MCMC is being reinitialized")
            self.mcmc.initialize()

        if not len(batch) >= self.batch_size:
            raise RuntimeError("Not enough samples that fulfill condition produced")

    def sample(self):
        return self.sample_batch()

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.sample_batch()

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


class LMDBDatasetSampler(object):
    def __init__(
        self,
        common_dataset: SamplesDataset,
        subset_distribution: Union[List[float], None],
        num_workers: int,
        batch_size: int,
        num_batches: Union[None, int] = None,
        infinite: bool = True,
        shuffle: bool = True,
    ) -> None:

        if not shuffle and (num_batches is None):
            num_batches = int(len(common_dataset) / batch_size)
            logger.info(
                f"Since shuffle is False and num_batches is None: num_batches set to {num_batches} in order to cover whole dataset"
            )
        elif shuffle and num_batches is None:
            num_batches = 1000

        if subset_distribution is None:
            dataset = ConcatDataset(common_dataset.datasets)
            if shuffle:
                sampler = RandomSampler(dataset, replacement=True)
            else:
                sampler = SequentialSampler(dataset)

            batch_sampler = BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=True,
            )

            _dl = DataLoader(
                dataset=dataset,
                worker_init_fn=lmdb_worker_init_fn,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
            )
        elif not shuffle and num_workers > 1:
            raise ValueError(
                "Only one worker implemented for no shuffle and subset distribution"
            )
        else:
            batch_sampler = MultipleDatasetsBatchSampler(
                batch_size=batch_size,
                common_dataset=common_dataset,
                distribution=subset_distribution,
                num_batches=num_batches,
                shuffle=shuffle,
            )

            _dl = DataLoader(
                common_dataset,
                worker_init_fn=lmdb_worker_init_fn,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
            )

        if infinite:
            self.dl = ExtendedDLIterator(_dl)
            self.sample = lambda device: self.dl.sample(device)
        else:
            self.dl = _dl.__iter__()
            self.sample = lambda device: self.dl.__next__().to(device)


class MCMCPSampler(object):
    def __init__(
        self,
        sampler_configs: List[MCMCSamplerConfig],
        batch_size: int,
        elements_per_dataset: int,
        subset_distribution: List[float],
        num_workers: int,
        shuffle: bool = True,
        infinite: bool = True,
    ) -> None:

        self.batch_size = batch_size
        mcmc_samplers = [MCMCSampler(**dict(conf)) for conf in sampler_configs]

        self.lmdb_pool = SamplesDataset(
            samplers=mcmc_samplers,
            refresh_rate=0,
            elements_per_dataset=elements_per_dataset,
            min_fill_level=1.0,
        )

        self.sampler = LMDBDatasetSampler(
            common_dataset=self.lmdb_pool,
            subset_distribution=subset_distribution,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            infinite=infinite,
        )

    def sample(self, device):
        return self.sampler.sample(device)

    @property
    def dataset(self):
        return self.lmdb_pool
