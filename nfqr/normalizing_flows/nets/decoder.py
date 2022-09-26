from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor, nn

from nfqr.normalizing_flows.nets.utils import (
    Activation,
    LayerNormalization,
    LayerNormalizationConfig,
)
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

from .utils import View

logger = create_logger(__name__)

DECODER_REGISTRY = StrRegistry("decoder")


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


@DECODER_REGISTRY.register("mlp")
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


class MLPDecoderConfig(BaseModel):

    net_hidden: Optional[List[int]]
    norm_configs: Optional[List[Union[LayerNormalizationConfig, None]]]
    n_groups: Optional[int] = 1


class DecoderConfig(BaseModel):

    decoder_type: str
    specific_decoder_config: MLPDecoderConfig
