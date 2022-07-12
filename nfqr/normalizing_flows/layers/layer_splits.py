from typing import List, Union

import numpy as np
import torch
from pydantic import BaseModel
from pyparsing import Optional

from nfqr.registry import StrRegistry

SPLIT_TYPES_REGISTRY = StrRegistry("split_types")


class LayerSplit(object):
    def __init__(
        self,
        dim,
        num_layers,
        split_type=Union[str, None],
        safe_guard: bool = True,
        **split_kwargs
    ):

        self.num_layers = num_layers
        self.dim = dim
        self.split_kwargs = split_kwargs

        if split_type == "n_transforms":
            self.split_fn = self.n_transforms_mask
        if split_type == "checkerboard":
            self.split_fn = self.checkerboard_mask

        if safe_guard:
            self.check_all_transformed()

    @classmethod
    def n_transforms(cls, dim, num_layers, n, safe_guard=True):
        if n is None:
            raise ValueError("Need n for n_transforms split")

        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="n_transforms",
            n=n,
        )

    @classmethod
    def checkerboard(cls, dim, num_layers, safe_guard=True):
        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="checkerboard",
        )

    def __iter__(self):

        for idx in range(self.num_layers):
            yield self.split_fn(size=self.dim[0], mask_num=idx, **self.split_kwargs)

    @staticmethod
    def checkerboard_mask(size, mask_num, **kwargs):
        # rewrite, this just works in 1d
        mask = (((torch.ones(size)).cumsum(dim=-1) + mask_num) % 2).bool()
        return mask, ~mask

    @staticmethod
    def n_transforms_mask(size, mask_num, n, **kwargs):

        mask = torch.zeros(size).bool()
        mask[mask_num :: int(np.ceil(size / n))] = True

        return ~mask, mask

    def check_all_transformed(self):

        acc_mask = torch.zeros(self.dim).bool()

        for conditioner_mask, transformed_mask in self:
            if conditioner_mask.any():
                acc_mask |= transformed_mask

        if acc_mask.sum() < self.dim[0] - 1:
            raise ValueError("Less than size-1 inputs transformed")


SPLIT_TYPES_REGISTRY.register("checkerboard", LayerSplit.checkerboard)
SPLIT_TYPES_REGISTRY.register("n_transforms", LayerSplit.n_transforms)


class SplitTypeConfig(BaseModel):
    dim: List[int]
    num_layers: int
    n: Optional[int]


class LayerSplitConfig(BaseModel):

    split_type: SPLIT_TYPES_REGISTRY.enum
    specific_split_type_config: SplitTypeConfig
