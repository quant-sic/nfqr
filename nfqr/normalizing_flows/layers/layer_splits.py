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
        elif split_type == "n_transforms_close":
            self.split_fn = self.n_transforms_close_mask
        elif split_type == "checkerboard":
            self.split_fn = self.checkerboard_mask
        elif split_type == "kernel":
            self.split_fn = self.kernel_mask

        if safe_guard:
            self.check_all_transformed()

    @classmethod
    def n_transforms(
        cls,
        dim,
        num_layers,
        n_transformed,
        num_offset,
        exclude_nn_of_transformed=False,
        safe_guard=True,
        **kwargs
    ):
        if n_transformed is None:
            raise ValueError("Need n for n_transforms split")

        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="n_transforms",
            n_transformed=n_transformed,
            num_offset=num_offset,
            exclude_nn_of_transformed=exclude_nn_of_transformed,
        )

    @classmethod
    def kernel(
        cls,
        dim,
        num_layers,
        kernel_size,
        dilation,
        stride,
        departmentalization,
        num_offset,
        safe_guard=True,
        **kwargs
    ):
        if any(v is None for v in (kernel_size, dilation, stride)):
            raise ValueError("Need kernel_size,dilation,stride for kernel split")

        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="kernel",
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            departmentalization=departmentalization,
            num_offset=num_offset,
        )

    @classmethod
    def n_transforms_close(
        cls,
        dim,
        num_layers,
        n_transformed,
        stride,
        num_offset,
        safe_guard=True,
        **kwargs
    ):
        if any(v is None for v in (n_transformed, stride)):
            raise ValueError("Need n and stride for n_transforms_close split")

        return cls(
            dim=dim,
            num_layers=num_layers,
            safe_guard=safe_guard,
            split_type="n_transforms_close",
            n_transformed=n_transformed,
            stride=stride,
            num_offset=num_offset,
        )

    @classmethod
    def checkerboard(cls, dim, num_layers, num_offset, safe_guard=True, **kwargs):
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
    def n_transforms_mask(
        size, mask_num, n_transformed, exclude_nn_of_transformed, **kwargs
    ):

        mask = torch.zeros(size).bool()
        step_size = int(np.ceil(size / n_transformed))

        transformed_idx = (torch.arange(n_transformed) * step_size + mask_num) % size
        mask[transformed_idx] = True

        if exclude_nn_of_transformed:
            exclude_mask = torch.roll(mask, shifts=-1, dims=0) | torch.roll(
                mask, shifts=1, dims=0
            )
        else:
            exclude_mask = torch.zeros(size).bool()

        assert not (
            mask & exclude_mask
        ).any(), "Exclude mask and transformed mask overlap"

        return ~(mask | exclude_mask), mask

    @staticmethod
    def n_transforms_close_mask(size, mask_num, n_transformed, stride, **kwargs):

        mask = torch.zeros(size).bool()

        transformed_idx = (torch.arange(n_transformed) + mask_num * stride) % size
        mask[transformed_idx] = True

        return ~mask, mask

    @staticmethod
    def kernel_mask(
        size, mask_num, kernel_size, stride, dilation, departmentalization, **kwargs
    ):

        if stride > 1 and departmentalization:
            raise ValueError("For departmentalization stride cannot be greater than 1")

        mask = torch.zeros(size).bool()

        department = int(mask_num / dilation)
        department_step = kernel_size * dilation
        extra_step = (
            mask_num * stride
            if not departmentalization
            else department * department_step + mask_num % dilation
        )

        transformed_idx = (torch.arange(kernel_size) * dilation + extra_step) % size
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
SPLIT_TYPES_REGISTRY.register("n_transforms_close", LayerSplit.n_transforms_close)
SPLIT_TYPES_REGISTRY.register("kernel", LayerSplit.kernel)


class SplitTypeConfig(BaseModel):
    n_transformed: Optional[int]
    kernel_size: Optional[int]
    dilation: Optional[int]
    stride: Optional[int]
    departmentalization: Optional[bool]
    exclude_nn_of_transformed: Optional[bool] = False


class LayerSplitConfig(BaseModel):

    split_type: str
    specific_split_type_config: Optional[SplitTypeConfig]
