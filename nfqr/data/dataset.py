from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from nfqr.utils.misc import create_logger

logger = create_logger(__name__)


class LmdbDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        super(LmdbDataset).__init__()

        dataset_file_path = Path(dataset_path)

        if not dataset_file_path.is_dir():
            raise ValueError(
                "Parameter 'dataset_path' needs to be point to a directory"
            )

        self.env = lmdb.open(
            str(dataset_file_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        logger.info("Initialising LmdbDataset")

        with self.env.begin(write=False) as txn:

            # assert that meta is not in there!!!
            self.length = txn.stat()["entries"] - 1
            # self.keys = [key for key, _ in txn.cursor() if not key=="meta"]

    def __getitem__(self, index: int) -> torch.Tensor:

        with self.env.begin(write=False) as txn:
            bytes = txn.get(str(index).encode("ascii"))

        item = torch.from_numpy(np.frombuffer(bytes, dtype=np.float32).copy())

        return item

    def __len__(self) -> int:
        return self.length
