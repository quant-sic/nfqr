import torch
from nfqr.normalizing_flows.nets.utils import Roll
from torch import nn

class CoordLayer(nn.Module):
    def __init__(self, mode, conditioner_mask, transformed_mask):
        super().__init__()

        added_channels_list = []
        added_transformations = []
        if "shift" in mode or "relative_position" in mode:
            if not transformed_mask.sum().item() == 1:
                raise ValueError("mode shift can only be used for n_transformed == 1 ")

        if "shift" in mode:
            transformed_position = (
                torch.arange(len(transformed_mask))[transformed_mask]
            ).item()
            added_transformations += [Roll(shifts=-transformed_position)]

        if "abs_position" in mode:
            added_channels_list += [
                (
                    torch.arange(len(conditioner_mask))[conditioner_mask]
                    / (len(conditioner_mask) - 1)
                )
                * 2
                - 1
            ]

        if "rel_position" in mode:
            # can only be one position
            transformed_position = (
                torch.arange(len(transformed_mask))[transformed_mask]
            ).item()
            added_channels_list += [
                (
                    2
                    * torch.min(
                        abs(torch.arange(len(conditioner_mask)) - transformed_position),
                        abs(
                            reversed(torch.arange(len(conditioner_mask)))
                            + transformed_position
                            + 1
                        ),
                    )[conditioner_mask]
                    / len(conditioner_mask)
                )
                * 2
                - 1
            ]

        self.added_channels = nn.parameter.Parameter(
            torch.stack(added_channels_list, dim=0), requires_grad=False
        )
        self.n_added_channels = len(added_channels_list)

    def forward(self, x):
        x_added_channels = torch.cat(
            (x, self.added_channels.expand(x.shape[0], *self.added_channels.shape)),
            dim=1,
        )

        return x_added_channels