from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
import torch
from scipy import stats


@dataclass
class Metrics:

    metrics_dict: Dict = field(default_factory=lambda: defaultdict(list))

    def add_batch_wise(
        self,
        _metrics_dict: Dict[str, Union[int, float]],
    ):

        for _key, _value in _metrics_dict.items():
            if isinstance(_value, torch.Tensor):
                _value = _value.item()
            self.metrics_dict[_key] += [_value]

    def last_slope(self, key: str, window_length: int):

        data = self.metrics_dict[key][-window_length:]
        slope = stats.linregress(np.arange(len(data)), data).slope

        slope_per_window = slope * len(data)
        return slope_per_window

    def last_mean(self, key: str, window_length: int):

        data = self.metrics_dict[key][-window_length:]
        return sum(data) / len(data)

    def last_fluctuation_around_linear(self, key: str, window_length: int):

        data = self.metrics_dict[key][-window_length:]
        linear_trend = stats.linregress(np.arange(len(data)), data)

        flucuation = np.std(
            data - (linear_trend.intercept + np.arange(len(data)) * linear_trend.slope)
        )

        return flucuation
