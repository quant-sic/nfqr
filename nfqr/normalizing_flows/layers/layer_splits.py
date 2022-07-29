from typing import List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel

from nfqr.registry import StrRegistry

SPLIT_TYPES_REGISTRY = StrRegistry("split_types")


class LayerSplit(object):
    def __init__(
        self,
        dim,
        num_layers,
        split_type=Union[str, None],
        num_offset: int = 0,
        safe_guard: bool = True,
        **split_kwargs
    ):

        self.num_layers = num_layers
        self.dim = dim
        self.split_kwargs = split_kwargs
        self.num_offset = num_offset

        if len(self.dim) > 1:
            raise ValueError("Splits not yet implemented for dim>1")

        if split_type == "n_transforms":
            self.split_fn = self.n_transforms_mask

        if split_type == "checkerboard":
            self.split_fn = self.checkerboard_mask

        if safe_guard:
            self.check_all_transformed()

    @classmethod
    def n_transforms(cls, dim, num_layers, n_transformed, num_offset, safe_guard=True):
        if n_transformed is None:
            raise ValueError("Need n for n_transforms split")

        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="n_transforms",
            n_transformed=n_transformed,
            num_offset=num_offset,
        )

    @classmethod
    def checkerboard(cls, dim, num_layers, num_offset, safe_guard=True):
        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="checkerboard",
            num_offset=num_offset,
        )

    def __iter__(self):

        for idx in range(self.num_layers):
            yield self.split_fn(
                size=self.dim[0], mask_num=idx + self.num_offset, **self.split_kwargs
            )

    @staticmethod
    def checkerboard_mask(size, mask_num, **kwargs):
        # rewrite, this just works in 1d
        mask = (((torch.ones(size)).cumsum(dim=-1) + mask_num) % 2).bool()
        return mask, ~mask

    @staticmethod
    def n_transforms_mask(size, mask_num, n_transformed, **kwargs):

        mask = torch.zeros(size).bool()
        step_size = int(np.ceil(size / n_transformed))

        transformed_idx = (torch.arange(n_transformed) * step_size + mask_num) % size
        mask[transformed_idx] = True

        return ~mask, mask

    def check_all_transformed(self):

        acc_mask = torch.zeros(self.dim).bool()

        for conditioner_mask, transformed_mask in self:
            if conditioner_mask.any():
                acc_mask |= transformed_mask

        if acc_mask.sum() < self.dim[0] - 1:
            raise ValueError("Less than size-1 inputs transformed")

    @property
    def all_conditioners_equal_in(self):

        num_in = []

        for conditioner_mask, _ in self:
            num_in += [conditioner_mask.sum().item()]

        return len(set(num_in)) == 1

    @property
    def all_conditioners_equal_out(self):

        num_out = []

        for _, transformed_mask in self:
            num_out += [transformed_mask.sum().item()]

        return len(set(num_out)) == 1

    @property
    def all_conditioners_equal_in_out(self):
        return self.all_conditioners_equal_in and self.all_conditioners_equal_out


SPLIT_TYPES_REGISTRY.register("checkerboard", LayerSplit.checkerboard)
SPLIT_TYPES_REGISTRY.register("n_transforms", LayerSplit.n_transforms)


class SplitTypeConfig(BaseModel):
    n_transformed: Optional[int]


class LayerSplitConfig(BaseModel):

    split_type: SPLIT_TYPES_REGISTRY.enum
    specific_split_type_config: Optional[SplitTypeConfig]
