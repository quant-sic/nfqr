from functools import cached_property
from typing import List, Literal, Union

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
from nfqr.target_systems.rotor import (
    ROTOR_TRAJECTORIES_REGISTRY,
    RotorTrajectorySamplerConfig,
)
from nfqr.utils import create_logger

from .config import ConditionConfig, TrajectorySamplerConfig

logger = create_logger(__name__)


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


class TrajectorySampler(object):
    def __init__(
        self,
        trajectory_sampler_config: Union[MCMCConfig, RotorTrajectorySamplerConfig],
        condition_config: ConditionConfig,
        sampler_batch_size: int,
        num_batches: int = 1,
    ) -> None:
        self.num_batches = num_batches

        if isinstance(trajectory_sampler_config, MCMCConfig):
            self.sampler = MCMC_REGISTRY[trajectory_sampler_config.mcmc_alg][
                trajectory_sampler_config.mcmc_type
            ](**dict(trajectory_sampler_config))
            self.sampler.initialize()
            self.sample_batch = self._sample_batch_mcmc

        elif isinstance(trajectory_sampler_config, RotorTrajectorySamplerConfig):
            self.sampler = ROTOR_TRAJECTORIES_REGISTRY[trajectory_sampler_config.traj_type](
                **dict(trajectory_sampler_config)
            )
            self.sample_batch = self._sample_batch_traj

        else:
            raise ValueError(f"Unkown Sampler config type {type(trajectory_sampler_config)}")

        self.condition = SampleCondition(**dict(condition_config))
        self.batch_size = sampler_batch_size

    @property
    def sampler_specs(self):
        return {**self.sampler.data_specs, "condition": str(self.condition)}

    def _sample_batch_mcmc(self, max_batch_repetitions=10, n_times_reinit=3):

        batch = []

        for _ in range(n_times_reinit):
            for _ in range(self.batch_size * max_batch_repetitions):

                self.sampler.step(record_observables=False)
                                
                if self.condition.evaluate(self.sampler.current_config):
                    batch += [self.sampler.current_config.detach().clone()]
                    if len(batch) >= self.batch_size:
                        return torch.concat(batch, dim=0)

            logger.info("MCMC is being reinitialized")
            self.sampler.initialize()

        if not len(batch) >= self.batch_size:
            raise RuntimeError("Not enough samples that fulfill condition produced")

    def _sample_batch_traj(self, max_batch_repetitions=10):

        batch = []

        for _ in range(self.batch_size * max_batch_repetitions):

            with torch.no_grad():
                config = self.sampler.sample(device="cpu")

            if self.condition.evaluate(config):
                batch += [config.detach().clone()]
                if len(batch) >= self.batch_size:
                    return torch.concat(batch, dim=0)

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

            self._dl = DataLoader(
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

            self._dl = DataLoader(
                common_dataset,
                worker_init_fn=lmdb_worker_init_fn,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
            )

        self.infinite = infinite

    def __len__(self):
        if self.infinite:
            raise RuntimeError("Sampler has infinite length")
        else:
            return len(self._dl)

    def __iter__(self):
        if self.infinite:
            return ExtendedDLIterator(self._dl)
        else:
            return self._dl.__iter__()


class PSampler(object):
    def __init__(
        self,
        trajectory_sampler_configs: List[TrajectorySamplerConfig],
        batch_size: int,
        elements_per_dataset: int,
        subset_distribution: List[float],
        num_workers: int,
        num_batches: int = None,
        shuffle: bool = True,
        infinite: bool = True,
    ) -> None:

        self.batch_size = batch_size

        mcmc_samplers = [TrajectorySampler(**dict(conf)) for conf in trajectory_sampler_configs]

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
            num_batches=num_batches,
        )

    @cached_property
    def sampler_iter(self):
        return self.sampler.__iter__()

    def sample(self, device):
        return self.sampler_iter.__next__().to(device)

    @property
    def dataset(self):
        return self.lmdb_pool

    def __iter__(self):
        yield from self.sampler.__iter__()

    def __len__(self):
        return len(self.sampler)
