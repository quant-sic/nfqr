from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from pydantic import BaseModel, validator
from torch import Tensor, nn

from nfqr.registry import StrRegistry

NET_REGISTRY = StrRegistry("nets")


@NET_REGISTRY.register("mlp")
class MLP(nn.Module):
    """a simple 4-layer MLP"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        out_channels: int,
        net_hidden: List[int],
        activation: str = "mish",
        **kwargs,
    ) -> None:
        super().__init__()

        if activation == "leaky_relu":
            activation_function = nn.LeakyReLU
        elif activation == "mish":
            activation_function = nn.Mish
        else:
            raise ValueError("Unknown Activation Function")

        modules = nn.ModuleList()

        sizes = [in_size] + net_hidden + [out_size * out_channels]

        for i in range(len(sizes) - 1):
            modules.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                modules.append(activation_function())

        modules.append(View([-1, out_size, out_channels]))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


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


@NET_REGISTRY.register("cnn")
class CNN(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        out_channels: int,
        net_hidden: List[int],
        pooling_types: Union[None, List[Union[None, Literal["avg", "max"]]]] = None,
        pooling_sizes: Union[None, List[Union[None, int]]] = None,
        activation: str = "mish",
        **kwargs,
    ) -> None:
        super().__init__()

        if activation == "leaky_relu":
            activation_function = nn.LeakyReLU
        elif activation == "mish":
            activation_function = nn.Mish
        else:
            raise ValueError("Unknown Activation Function")

        modules = nn.ModuleList()

        if in_size != out_size:
            modules.append(nn.Linear(in_size, out_size))
            modules.append(activation_function())

        modules.append(View([-1, 1, out_size]))

        net_hidden = [1] + net_hidden
        for layer_idx, (in_, out_) in enumerate(zip(net_hidden[:-1], net_hidden[1:])):
            modules.append(
                nn.Conv1d(
                    in_channels=in_,
                    out_channels=out_,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    padding_mode="circular",
                )
            )
            modules.append(activation_function())

            if pooling_types is not None and pooling_sizes is not None:
                pooling_layer = self.pooling_layer(
                    pooling_types[layer_idx], pooling_sizes[layer_idx]
                )

                if pooling_layer is not None:
                    modules.append(pooling_layer)

        modules.append(View([-1, net_hidden[-1] * out_size]))
        modules.append(nn.Linear(net_hidden[-1] * out_size, out_channels * out_size))
        modules.append(View([-1, out_size, out_channels]))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def pooling_layer(specifier, size):
        if specifier == "none":
            return None
        elif specifier == "avg":
            return nn.AdaptiveAvgPool1d(size)
        elif specifier == "max":
            return nn.AdaptiveMaxPool1d(size)
        else:
            raise ValueError(f"Unknown pooling type {specifier}")


class Roll(nn.Module):
    def __init__(self, shifts, dims=-1):
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x):
        return x.roll(shifts=self.shifts, dims=self.dims)


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


class AtrousConvolution(nn.Module):
    def __init__(self,dilations,in_channels,out_channels,
                        kernel_size,
                        stride,
                        padding,
                        padding_mode,
                        groups) -> None:
        super().__init__()

        self.convs = nn.ModuleList()

        for d in dilations:
            self.convs.append(
                nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        groups=groups,
                        dilation=d
                    ))

    def forward(self,x):
        return torch.cat([conv(x) for conv in self.convs],dim=1)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        n_channels: List[int],
        residual: bool,
        activation_specifier: str,
        norm_configs: List[Union[LayerNormalizationConfig, None]],
        kernel_sizes: Union[List[int], None],
        concat_input: bool = False,
        n_groups: int = 1,
        dilations:List[int]=[1]
    ) -> None:
        super().__init__()

        self.layers = []

        n_channels_list = [in_channels] + n_channels
        norm_configs = (
            [None] * len(n_channels) if norm_configs is None else norm_configs
        )
        kernel_sizes = [3] * len(n_channels) if kernel_sizes is None else kernel_sizes
            
        for in_, out_, norm_config, kernel_size_ in zip(
            n_channels_list[:-1], n_channels_list[1:], norm_configs, kernel_sizes
        ):  
            self.layers.append(AtrousConvolution(dilations=dilations,in_channels=in_,
                    out_channels=out_,
                    kernel_size=kernel_size_,
                    stride=1,
                    padding=kernel_size_ // 2,
                    padding_mode="circular",
                    groups=n_groups)

            )

            self.layers.append(Activation(activation_specifier=activation_specifier))

            if norm_config is not None:
                self.layers.append(
                    LayerNormalization(
                        **dict(norm_config), out_channel=out_, out_size=dim
                    )
                )

        self.residual = residual
        if residual:
            if n_channels[-1] != in_channels:
                raise ValueError(
                    f"In channels {in_channels} do not match out channels {n_channels[-1]}, so residual construction not possible"
                )

        self.concat_input = concat_input
        self._out_channels = n_channels[-1]
        if self.concat_input:
            self._out_channels += in_channels

        self.net = nn.Sequential(*self.layers)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):

        x_net = self.net(x)

        if self.residual:
            x_out = x + x_net
        else:
            x_out = x_net

        if self.concat_input:
            x_out = torch.cat([x, x_out], dim=1)

        return x_out


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


class EncoderBlockConfig(BaseModel):

    n_channels: Union[List[int], int]
    residual: bool = False
    activation_specifier: str = "mish"
    norm_configs: Union[List[Union[LayerNormalizationConfig, None]], None] = None
    kernel_sizes: Union[List[int], None] = None
    concat_input: bool = False
    n_groups: int = 1
    dilations:List[int]=[1]

    @validator("n_channels", pre=True)
    @classmethod
    def to_list(cls, v):
        if isinstance(v, int):
            return [v]
        else:
            return v


@NET_REGISTRY.register("cnn_encoder")
class CNNEncoder(nn.Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        in_channels,
        block_configs: List[EncoderBlockConfig],
        pooling_configs: List[Union[PoolingConfig, None]],
        coord_layer_specifier: Union[bool, str, None] = "rel_position",
        **kwargs,
    ) -> None:
        super().__init__()

        blocks = []
        pooling_configs = (
            [None] * len(block_configs) if pooling_configs is None else pooling_configs
        )
        in_size = conditioner_mask.sum().item()

        if coord_layer_specifier is not None:
            coord_layer = CoordLayer(
                coord_layer_specifier,
                conditioner_mask=conditioner_mask,
                transformed_mask=transformed_mask,
            )
            blocks.append(coord_layer)
            in_channels += coord_layer.n_added_channels

        for block_config, pooling_config in zip(block_configs, pooling_configs):

            encoder_block = EncoderBlock(
                **dict(block_config),
                in_channels=in_channels,
                dim=in_size,
            )

            blocks.append(encoder_block)
            in_channels = encoder_block.out_channels

            if pooling_config is not None:
                pooling = Pooling(**dict(pooling_config))
                blocks.append(pooling)
                in_size = pooling.out_size

        self.net = nn.Sequential(*blocks)

        self._dim_out = in_size
        self._out_channels = in_channels

    @property
    def dim_out(self):
        return self._dim_out

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.net(x)


class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, mask: Tensor = None
    ) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(dim_in, dim_out, bias)
        self._mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        self._mask = m

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)


def grouped_conv_mask_for_linear(dim_in, dim_out, n_channels):

    mask = torch.zeros(dim_out, n_channels, dim_in)

    for g in range(dim_out):

        g_first, g_last = g * int(n_channels / dim_out), (g + 1) * int(
            n_channels / dim_out
        )
        mask[g, g_first:g_last, :] = 1

    return mask.view(dim_out, n_channels * dim_in)


@NET_REGISTRY.register("mlp_decoder")
class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        in_channels: int,
        out_size: int,
        out_channels: int,
        net_hidden: List[int],
        norm_configs: Union[List[Union[LayerNormalizationConfig, None]], None] = None,
        activation_specifier: str = "mish",
        n_groups: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(View([-1, in_size * in_channels]))

        if n_groups > 1 and len(net_hidden) > 0:
            raise ValueError("for hidden layers n_groups>1 undefined and ignored")

        sizes = [in_size * in_channels] + net_hidden + [out_size * out_channels]
        norm_configs = (
            [None] * len(net_hidden) if norm_configs is None else norm_configs
        )
        norm_configs += [None]

        for idx, (in_, out_, norm_config) in enumerate(
            zip(sizes[:-1], sizes[1:], norm_configs)
        ):
            if n_groups == 1:
                layers.append(nn.Linear(in_, out_))
            else:
                layers.append(
                    MaskedLinear(
                        in_,
                        out_,
                        mask=grouped_conv_mask_for_linear(in_, out_, in_channels),
                    )
                )

            if idx != len(sizes) - 2:
                layers.append(Activation(activation_specifier=activation_specifier))

            if norm_config is not None:
                layers.append(
                    LayerNormalization(
                        **dict(norm_config), out_channel=out_, out_size=1
                    )
                )

        layers.append(View([-1, out_size, out_channels]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NetConfig(BaseModel):

    net_type: str
    net_hidden: Optional[List[int]]
    norm_configs: Optional[List[Union[LayerNormalizationConfig, None]]]
    coord_layer_specifier: Optional[
        Literal["rel_position", "abs_position", "rel_position+abs_position"]
    ]
    n_groups: Optional[int] = 1

    block_configs: Optional[List[EncoderBlockConfig]]
    pooling_configs: Optional[List[Union[PoolingConfig, None]]] = None
