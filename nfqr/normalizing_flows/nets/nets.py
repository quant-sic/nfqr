from typing import List, Literal,Union

from pydantic import BaseModel
from torch import nn

from nfqr.registry import StrRegistry

NET_REGISTRY = StrRegistry("nets")


class NetConfig(BaseModel):

    net_type: str
    net_hidden: List[int]


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
        **kwargs
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
        pooling_types:Union[None,List[Union[None,Literal["avg","max"]]]] = None,
        pooling_sizes:Union[None,List[Union[None,int]]] = None,
        activation: str = "mish",
        **kwargs
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
        for layer_idx,(in_, out_) in enumerate(zip(net_hidden[:-1], net_hidden[1:])):
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
                pooling_layer = self.pooling_layer(pooling_types[layer_idx],pooling_sizes[layer_idx])

                if not pooling_layer is None:
                    modules.append(pooling_layer)

        modules.append(View([-1, net_hidden[-1] * out_size]))
        modules.append(nn.Linear(net_hidden[-1] * out_size, out_channels * out_size))
        modules.append(View([-1, out_size, out_channels]))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def pooling_layer(specifier,size):
        if specifier=="none":
            return None
        elif specifier == "avg":
            return nn.AdaptiveAvgPool1d(size)
        elif specifier == "max":
            return nn.AdaptiveMaxPool1d(size)
        else:
            raise ValueError(f"Unknown pooling type {specifier}")
