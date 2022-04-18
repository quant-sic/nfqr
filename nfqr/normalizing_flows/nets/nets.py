from typing import List

from torch import nn


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


class FlowCNN(nn.Module):
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

        if in_size != out_size:
            modules.append(nn.Linear(in_size, out_size))
            modules.append(activation_function())

        modules.append(View([-1, 1, out_size]))

        net_hidden = [1] + net_hidden
        for in_, out_ in zip(net_hidden[:-1], net_hidden[1:]):
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

        modules.append(View([-1, net_hidden[-1] * out_size]))
        modules.append(nn.Linear(net_hidden[-1] * out_size, out_channels * out_size))
        modules.append(View([-1, out_size, out_channels]))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
