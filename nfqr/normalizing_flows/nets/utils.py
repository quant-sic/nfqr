from torch import nn
from typing import Literal

from pydantic import BaseModel

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Permute(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)


class Roll(nn.Module):
    def __init__(self, shifts, dims=-1):
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x):
        return x.roll(shifts=self.shifts, dims=self.dims)

class Activation(nn.Module):
    def __init__(self, activation_specifier) -> None:
        super().__init__()

        if activation_specifier == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_specifier == "mish":
            self.activation = nn.Mish()
        else:
            raise ValueError("Unknown Activation Function")

    def forward(self, x):
        return self.activation(x)


class LayerNormalization(nn.Module):
    def __init__(self, norm_type, out_channel, out_size, norm_affine) -> None:
        super().__init__()

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channel, affine=norm_affine)
        if norm_type == "layer":
            self.norm = nn.LayerNorm(
                normalized_shape=(out_channel, out_size), elementwise_affine=norm_affine
            )

    def forward(self, x):
        return self.norm(x)


class LayerNormalizationConfig(BaseModel):

    norm_type: Literal["batch", "layer"]
    norm_affine: bool = True


class Pooling(nn.Module):
    def __init__(self, pooling_type, pooling_out_size) -> None:
        super().__init__()

        if pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool1d(output_size=pooling_out_size)
        elif pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(output_size=pooling_out_size)
        else:
            raise ValueError(f"Unknown pooling type {pooling_type}")

        self._out_size = pooling_out_size

    @property
    def out_size(self):
        return self._out_size

    def forward(self, x):
        return self.pooling(x)


class PoolingConfig(BaseModel):

    pooling_type: Literal["max", "avg"]
    pooling_out_size: int