import torch
from strenum import StrEnum


class SPLIT_TYPES(StrEnum):
    checkerboard = "checkerboard"
    autoregressive = "autoregressive"
    autoregressive_2 = "autoregressive_2"


def checkerboard_mask(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask = (((torch.ones(size)).cumsum(-1) + mask_config) % 2).bool()
    return mask, ~mask


def autoregressive_mask(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask_conditioner = torch.ones(size).cumsum(-1) <= mask_config

    mask_transformed = torch.zeros(size).bool()
    mask_transformed[mask_config] = True

    return mask_conditioner, mask_transformed

def autoregressive_mask_2(size, mask_config, **kwargs):
    # rewrite, this just works in 1d
    mask_transformed = torch.zeros(size).bool()
    mask_transformed[mask_config] = True

    return ~mask_transformed, mask_transformed

def generate_splits(split_type, num_layers, size):

    if split_type == "checkerboard":
        split_fn = checkerboard_mask
    elif split_type == "autoregressive":
        split_fn = autoregressive_mask
    elif split_type == "autoregressive_2":
        split_fn = autoregressive_mask_2
    else:
        raise ValueError("mask type not implemented")

    for layer_idx in range(num_layers):
        conditioner_mask, transformed_mask = split_fn(size=size, mask_config=layer_idx)
        yield conditioner_mask, transformed_mask
