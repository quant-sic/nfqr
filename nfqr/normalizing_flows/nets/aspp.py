from typing import List

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn


class AtrousConvolution(nn.Module):
    def __init__(
        self,
        dilations,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_mode,
        groups,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()

        for d in dilations:
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=((kernel_size - 1) * d) // 2,
                    padding_mode=padding_mode,
                    groups=groups,
                    dilation=d,
                )
            )

    def forward(self, x):
        return torch.stack([conv(x).unsqueeze(2) for conv in self.convs], dim=2).view(
            x.shape[0], -1, x.shape[-1]
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-1:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="linear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(
        self,
        conditioner_mask,
        in_channels: int,
        out_channels: int,
        n_aspp: int = 1,
        internal_dim: int = 32,
        aspp_atrous_rates: List[int] = [1, 2, 3],
        **kwargs
    ) -> None:

        aspp_layers = []
        aspp_layers.append(
            ASPP(in_channels, atrous_rates=aspp_atrous_rates, out_channels=internal_dim)
        )
        for _ in range(n_aspp - 1):
            aspp_layers.append(
                ASPP(
                    in_channels=internal_dim,
                    atrous_rates=aspp_atrous_rates,
                    out_channels=internal_dim,
                )
            )

        super().__init__(
            *aspp_layers,
            nn.Conv1d(internal_dim, internal_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(internal_dim),
            nn.ReLU(),
            nn.Conv1d(internal_dim, out_channels, 1),
        )

        self.out_channels = out_channels
        self.dim_out = conditioner_mask.sum().item()


class DeepLabHeadConfig(BaseModel):

    n_aspp: int = 1
    internal_dim: int = 32
    aspp_atrous_rates: List[int]
    out_channels: int
