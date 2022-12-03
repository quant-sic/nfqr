import lmdb
from nfqr.globals import EXPERIMENTS_DIR,DATASETS_DIR
import shutil
from nfqr.utils import create_logger
import numpy as np

logger = create_logger(__name__)

if __name__ == "__main__":
    
    for n_elements in range(1000,250000,10000):
        logger.info(int(n_elements * 4 * 2)/1e9)
        
        if (DATASETS_DIR/"test").is_dir():
            shutil.rmtree(DATASETS_DIR/"test")

        env = lmdb.open(
            str(DATASETS_DIR/"test"),
            writemap=True
        )

        logger.info(int(n_elements * 4 * 2)/1e9)
