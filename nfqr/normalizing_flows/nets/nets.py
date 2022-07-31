from typing import List, Literal, Optional, Union

import torch
from pydantic import BaseModel, validator
from torch import nn

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
                    2*torch.min(
                        abs(torch.arange(len(conditioner_mask)) - transformed_position),
                        abs(
                            reversed(torch.arange(len(conditioner_mask)))
                            + transformed_position
                            + 1
                        ),
                    )[conditioner_mask]
                    / len(conditioner_mask)
                ) * 2 - 1
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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        n_channels: List[int],
        residual: bool,
        activation_specifier: str,
        norms: List[Union[str, None]],
        kernel_sizes: Union[List[int], None],
        concat_input:bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = Activation(activation_specifier=activation_specifier)

        n_channels_list = [in_channels] + n_channels
        norms = [None] * len(n_channels) if norms is None else norms
        kernel_sizes = [3] * len(n_channels) if kernel_sizes is None else kernel_sizes

        for in_, out_, norm_, kernel_size_ in zip(
            n_channels_list[:-1], n_channels_list[1:], norms, kernel_sizes
        ):
            self.layers.append(
                nn.Conv1d(
                    in_channels=in_,
                    out_channels=out_,
                    kernel_size=kernel_size_,
                    stride=1,
                    padding=kernel_size_ // 2,
                    padding_mode="circular",
                )
            )

            self.layers.append(self.activation)

            if norm_ == "batch":
                self.layers.append(nn.BatchNorm1d(out_))
            if norm_ == "layer":
                self.layers.append(nn.LayerNorm(normalized_shape=(out_, dim)))
            


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
            x_out = torch.cat([x,x_out],dim=1)

        return x_out

    
class EncoderBlockConfig(BaseModel):

    n_channels:Union[List[int],int]
    residual: bool = False
    activation_specifier: str = "mish"
    norms: Union[List[Union[str, None]], None] = None
    kernel_sizes: Union[List[int], None] = None
    concat_input:bool = False

    @validator("n_channels",pre=True)
    @classmethod
    def to_list(cls,v):
        if isinstance(v,int):
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
        pooling_sizes: List[Union[int, None]],
        coord_layer_specifier: Union[bool, str, None] = "rel_position",
        **kwargs,
    ) -> None:
        super().__init__()

        blocks = nn.ModuleList()
        pooling_sizes = (
            [None] * len(block_configs) if pooling_sizes is None else pooling_sizes
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

        for block_config, pooling_size_ in zip(block_configs, pooling_sizes):

            encoder_block = EncoderBlock(
                    **dict(block_config),
                    in_channels=in_channels,
                    dim=in_size,
                )

            blocks.append(encoder_block)
            in_channels = encoder_block.out_channels

            if pooling_size_ is not None:
                blocks.append(nn.AdaptiveMaxPool1d(pooling_size_))
                in_size = pooling_size_

        self.net = nn.Sequential(*blocks)

        self._dim_out = (
            [conditioner_mask.sum().item()]
            + list(filter(lambda s: s is not None, pooling_sizes))
        )[-1]
        self._out_channels = in_channels

    @property
    def dim_out(self):
        return self._dim_out

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.net(x)


@NET_REGISTRY.register("mlp_decoder")
class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        in_channels: int,
        out_size: int,
        out_channels: int,
        net_hidden: List[int],
        norms: Union[List[Union[str, None]], None] = None,
        activation_specifier: str = "mish",
        **kwargs
    ) -> None:
        super().__init__()

        self.activation = Activation(activation_specifier=activation_specifier)


        modules = nn.ModuleList()
        modules.append(View([-1, in_size * in_channels]))

        sizes = [in_size * in_channels] + net_hidden + [out_size * out_channels]
        norms = [None] * len(net_hidden) if norms is None else norms
        norms += [None]

        for idx,(in_,out_,norm_) in enumerate(zip(sizes[:-1],sizes[1:],norms)):
            modules.append(nn.Linear(in_, out_))

            if norm_ == "batch":
                self.layers.append(nn.BatchNorm1d(out_))
            if norm_ == "layer":
                self.layers.append(nn.LayerNorm(normalized_shape=(out_)))                

            if idx != len(sizes) - 2:
                modules.append(self.activation)


        modules.append(View([-1, out_size, out_channels]))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class NetConfig(BaseModel):

    net_type: str
    net_hidden: Optional[List[int]]
    norms:Optional[List[Union[str, None]]]
    coord_layer_specifier: Optional[
        Literal["rel_position", "abs_position", "rel_position+abs_position"]
    ]

    block_configs: Optional[List[EncoderBlockConfig]]
    pooling_sizes: Optional[Union[List[Union[int, None]], None]] = None
