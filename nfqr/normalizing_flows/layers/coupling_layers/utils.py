import torch
from strenum import StrEnum


class SPLIT_TYPES(StrEnum):
    checkerboard = "checkerboard"
    single_transforms = "single_transforms"


def checkerboard_mask(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask = (((torch.ones(size)).cumsum(-1) + mask_config) % 2).bool()
    return mask, ~mask


def single_transforms(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask_transformed = torch.zeros(size).bool()
    mask_transformed[mask_config] = True

    return ~mask_transformed, mask_transformed


def generate_splits(split_type, num_layers, size):

    if split_type == "checkerboard":
        split_fn = checkerboard_mask
    elif split_type == "single_transforms":
        split_fn = single_transforms
    else:
        raise ValueError("mask type not implemented")

    for layer_idx in range(num_layers):
        conditioner_mask, transformed_mask = split_fn(size=size, mask_config=layer_idx)
        yield conditioner_mask, transformed_mask
