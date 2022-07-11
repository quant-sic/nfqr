import torch
from strenum import StrEnum


class SPLIT_TYPES(StrEnum):
    checkerboard = "checkerboard"
    autoregressive = "autoregressive"


def checkerboard_mask(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask = (((torch.ones(size)).cumsum(-1) + mask_config) % 2).bool()
    return mask, ~mask


def generate_splits(split_type, num_layers, size):

    if split_type == "checkerboard":
        split_fn = checkerboard_mask
    else:
        raise ValueError("mask type not implemented")

    for layer_idx in range(num_layers):
        conditioner_mask, transformed_mask = split_fn(size=size, mask_config=layer_idx)
        yield conditioner_mask, transformed_mask
