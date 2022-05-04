import math

import torch


def get_iid_statistics(history):

    with torch.no_grad():
        count = history.shape[0]
        mean = history.mean()
        sq_mean = (history**2).mean()

        std = torch.sqrt(abs(sq_mean - mean**2))
        err = std / math.sqrt(count)

        return {"mean": mean.item(), "error": err.item()}
