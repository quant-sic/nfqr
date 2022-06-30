import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nfqr.data.condition import SampleCondition
from nfqr.globals import DATASETS_DIR
from nfqr.utils import NumpyEncoder, create_logger

logger = create_logger(__name__)


class LmdbDataset(Dataset):
    def __init__(self, dataset_path: Path, max_size: int) -> None:
        super(LmdbDataset).__init__()

        self.dataset_file_path = Path(dataset_path)
        self.max_size = max_size

        # 4 bytes per float, multiply by 5 for good measure
        self.env = lmdb.open(
            str(self.dataset_file_path),
            lock=False,
            readahead=False,
            meminit=True,
            map_size=int(self.max_size * 4 * 5),
        )

    def reinit_env(self):
        self.env.close()
        self.env = lmdb.open(
            str(self.dataset_file_path),
            lock=False,
            readahead=False,
            meminit=True,
            map_size=int(self.max_size * 4 * 5),
            readonly=True,
        )

    @property
    def keys(self):
        with self.env.begin(write=False) as txn:
            keys = list(map(lambda k: k[0].decode("ascii"), txn.cursor()))
        return keys

    @property
    def length(self):
        if not hasattr(self, "_length"):
            with self.env.begin(write=False) as txn:
                # excluding meta data
                self._length = txn.stat()["entries"]

                if "meta" in map(lambda k: k[0].decode("ascii"), txn.cursor()):
                    self._length -= 1

        return self._length

    @length.setter
    def length(self, v):
        self._length = v

    @property
    def meta(self):

        with self.env.begin(write=False) as txn:
            _meta = txn.get("meta".encode("ascii"))
            if _meta is None:
                logger.error("Meta has not been set yet")
                # shutil.rmtree(dset)
            else:
                _meta = _meta.decode("ascii")

        _meta_dic = json.loads(_meta)

        return _meta_dic

    @meta.setter
    def meta(self, v: Dict):

        with self.env.begin(write=True) as txn:
            meta = json.dumps(v, cls=NumpyEncoder)
            txn.put("meta".encode("ascii"), meta.encode("ascii"))

    def __getitem__(self, index: int) -> torch.Tensor:

        try:
            with self.env.begin(write=False) as txn:
                if isinstance(index, int):
                    bytes = txn.get(str(index).encode("ascii"))
                    item = torch.from_numpy(
                        np.frombuffer(bytes, dtype=np.float32).copy()
                    )

                elif isinstance(index, (list, np.ndarray, torch.Tensor)):
                    if isinstance(index, (np.ndarray, torch.Tensor)):
                        assert index.ndim == 1, "Invalid Index shape"

                    bytes = txn.getmulti(map(lambda i: str(i).encode("ascii"), index))
                    item = torch.from_numpy(
                        np.stack(
                            [
                                np.frombuffer(value, dtype=np.float32).copy()
                                for key, value in bytes
                            ],
                            axis=0,
                        )
                    )

                else:
                    raise ValueError(f"Unhandled Index type {type(index)}")
        except TypeError as e:
            logger.info(f"Type Error {e} for index {index}")
            raise

        return item

    def replace(
        self,
        index: Union[List[int], int, np.ndarray, torch.Tensor],
        samples: torch.Tensor,
    ) -> torch.Tensor:

        with self.env.begin(write=True) as txn:

            if isinstance(index, int) and samples.ndim == 1:
                bytes = txn.get(str(index).encode("ascii"))
                item = torch.from_numpy(np.frombuffer(bytes, dtype=np.float32).copy())
                txn.replace(
                    str(index).encode("ascii"),
                    samples.numpy().astype(np.float32).tobytes(),
                )

            elif isinstance(index, (list, np.ndarray, torch.Tensor)):
                if isinstance(index, (np.ndarray, torch.Tensor)):
                    assert index.ndim == 1, "Invalid Index shape"

                assert samples.ndim == 2, "Index and Samples shape do not match"

                item_list = []
                for idx, sample in zip(index, samples):
                    bytes = txn.get(str(idx).encode("ascii"))
                    item_list += [
                        torch.from_numpy(np.frombuffer(bytes, dtype=np.float32).copy())
                    ]
                    txn.put(
                        str(idx).encode("ascii"),
                        sample.numpy().astype(np.float32).tobytes(),
                    )

                item = torch.stack(item_list, dim=0)

        return item

    def append(self, samples: torch.Tensor) -> None:

        with self.env.begin(write=True) as txn:

            if samples.ndim == 1:

                txn.put(
                    str(len(self)).encode("ascii"),
                    samples.numpy().astype(np.float32).tobytes(),
                )

            elif samples.ndim == 2:

                for idx, sample in enumerate(samples):
                    txn.put(
                        str(len(self) + idx).encode("ascii"),
                        sample.numpy().astype(np.float32).tobytes(),
                    )
            else:

                raise ValueError(f"Invalid shape of samples:{samples.shape}")

        self.length += samples.shape[0]

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        self.env.close()

    def isequal(self, values_dict: dict, n_elements) -> bool:
        def _item_is_equal(k, v_other):
            v_meta = self.meta.get(k, "")

            if k == "condition":
                return SampleCondition.from_str_or_dict(v_meta) == SampleCondition.from_str_or_dict(
                    v_other
                )
            else:
                return v_meta == v_other

        return (
            all(_item_is_equal(k, v) for k, v in values_dict.items())
            and self.max_size >= n_elements
        )


def get_lmdb_dataset(values_dict, n_elements):

    dset_paths = list(sorted(DATASETS_DIR.glob("*")))
    max_size = n_elements * np.array(values_dict["dim"]).prod()
    for dset_path in tqdm(dset_paths, desc="Checking datasets"):

        try:
            lmdb_dataset = LmdbDataset(dset_path, max_size=max_size)
        except:
            logger.info(f"Dataset {dset_path.name} seems to be corrupt and is deleted")
            shutil.rmtree(dset_path)

        if lmdb_dataset.isequal(values_dict=values_dict, n_elements=n_elements):
            logger.info(f"Using existing dataset: {dset_path.name}")
            return lmdb_dataset

    dset_name = (
        list(
            map(
                str,
                list(
                    set(range(len(dset_paths) + 1))
                    - set(list(map(lambda path: int(path.name), dset_paths)))
                ),
            )
        )
        + ["0"]
    )[0]
    logger.info(f"Creating new dataset: {dset_name}")

    new_dataset = LmdbDataset(DATASETS_DIR / dset_name, max_size=max_size)
    new_dataset.meta = values_dict

    return new_dataset


# class HMCDataset(Dataset):
#     def __init__(
#         self,
#         action: Action,
#         lat_shape: List[int],
#         n_steps: int = 0,
#         equilibration_steps: int = 0,
#         des_acc_perc: float = 0.9,
#         save_interval: int = 10**9,
#         eps: float = 0.05,
#         traj_steps: int = 20,
#         observables: ObservableMeter = None,
#         bias: float = 0.0,
#         mode_dropping: bool = True,
#         model=None,
#         autotune_step: bool = True,
#         batch_size: int = 1,
#         initial_configs=None,
#         int_steps: int = 1,
#         condition=None,
#     ):

#         self.int_steps = int_steps
#         self.index_in_batch = 0
#         self.batch_size = batch_size
#         self.lat_shape = lat_shape
#         self.condition = condition

#         if observables is None:
#             observables = ObservableMeter({})

#         self.hmc = HMCMarkovChain(
#             action=action,
#             lat_shape=lat_shape,
#             n_steps=n_steps,
#             equilibration_steps=equilibration_steps,
#             des_acc_perc=des_acc_perc,
#             save_interval=save_interval,
#             eps=eps,
#             traj_steps=traj_steps,
#             observables=observables,
#             overrelax_freq=0,
#             bias=bias,
#             mode_dropping=mode_dropping,
#             model=model,
#             autotune_step=autotune_step,
#             batch_size=batch_size,
#             initial_configs=initial_configs,
#         )

#         # initialize config
#         self.config = self.hmc.initialize()

#         # equilibration steps
#         logger.info("starting with equilibration steps")
#         for _ in tqdm(range(equilibration_steps)):
#             self.config = self.hmc.step(self.config)

#     def sample(self, device):

#         batch = []
#         for _ in range(self.batch_size):
#             batch += [self.__getitem__()]

#         sample = torch.stack(batch, dim=0).to(device=device)
#         return sample

#     def __getitem__(self, item=None, max_tries=1000):

#         # Important to handle this
#         for n_try in range(max_tries):

#             if self.index_in_batch and (self.index_in_batch + 1) % self.batch_size == 0:

#                 for _ in range(self.int_steps):
#                     self.config = self.hmc.step(self.config)

#                 self.index_in_batch = 0
#             else:
#                 self.index_in_batch += 1

#             ret_config = self.config[[self.index_in_batch]]

#             if self.condition is None or self.condition(ret_config):
#                 break

#             if n_try >= max_tries - 1:
#                 self.config = self.hmc.initialize()
#                 ret_config = self.config[[self.index_in_batch]]

#         return ret_config.detach().clone().float().view(*self.lat_shape)

#     def __len__(self):
#         return int(1e7)


class SamplesDataset(Dataset):
    def __init__(
        self,
        samplers,
        refresh_rate: float,
        elements_per_dataset: int,
        min_fill_level: float,
        dataset_dir: Path = DATASETS_DIR,
    ) -> None:
        super(SamplesDataset).__init__()

        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.samplers = samplers

        self.datasets = [
            get_lmdb_dataset(_sampler.sampler_specs, n_elements=elements_per_dataset)
            for _sampler in samplers
        ]

        self.current_step = 0
        self.refresh_rate = refresh_rate
        self.elements_per_dataset = elements_per_dataset

        self.fill_datasets(min_fill_level=min_fill_level)

    def fill_datasets(self, min_fill_level: float = 1.0):

        for _sampler, _set in tqdm(zip(self.samplers, self.datasets)):

            for _ in tqdm(
                range(
                    max(
                        int(
                            np.ceil(
                                (self.elements_per_dataset * min_fill_level - len(_set))
                                / _sampler.batch_size
                            )
                        ),
                        0,
                    )
                )
            ):

                _set.append(_sampler.sample())

    def __getitem__(self, idx: Tuple[int, int]):

        self.current_step += 1

        # worker_info = torch.utils.data.get_worker_info()

        # if (worker_info is None or worker_info.id ==0) and (self.refresh_rate!=0 and self.current_step % int(np.round(max(1/(self.refresh_rate*getattr(worker_info,"num_workers",1)),1))) == 0):

        #     for _sampler,_set in zip(self.samplers,self.datasets):
        #         new_samples = _sampler.sample(self.sampler_batch_size)
        #         if len(_set)<self.elements_per_dataset:
        #             _set.append(new_samples)
        #         else:
        #             index = np.random.randint(0,high = len(_set)-1,size=self.sampler_batch_size)
        #             _set.replace(index,new_samples)

        #     return new_samples

        # else:
        #     return self.datasets[idx[0]][1]

        return self.datasets[idx[0]][idx[1]]

    def __len__(self):
        return sum(len(_set) for _set in self.datasets)


def lmdb_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    for _set in dataset.datasets:
        _set.reinit_env()
