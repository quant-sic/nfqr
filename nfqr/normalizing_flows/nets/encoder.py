from typing import List, Literal, Optional, Union

import torch
from pydantic import BaseModel, validator
from torch import nn

from nfqr.normalizing_flows.nets.utils import (
    Activation,
    LayerNormalization,
    LayerNormalizationConfig,
    Pooling,
    PoolingConfig,
)
from nfqr.registry import StrRegistry
from nfqr.utils import create_logger

from .aspp import AtrousConvolution, DeepLabHead, DeepLabHeadConfig
from .coord_conv import CoordLayer

logger = create_logger(__name__)

ENCODER_REGISTRY = StrRegistry("encoder")

ENCODER_REGISTRY.register("deep_lab_head", DeepLabHead)


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
        dilations: List[int] = [1],
    ) -> None:
        super().__init__()

        self.layers = []

        n_channels_list = [in_channels] + n_channels
        norm_configs = (
            [None] * len(n_channels) if norm_configs is None else norm_configs
        )
        kernel_sizes = [3] * len(n_channels) if kernel_sizes is None else kernel_sizes

        for idx, (in_, out_, norm_config, kernel_size_) in enumerate(
            zip(n_channels_list[:-1], n_channels_list[1:], norm_configs, kernel_sizes)
        ):
            if idx > 0:
                in_ *= len(dilations)



            self.layers.append(
                AtrousConvolution(
                    dilations=dilations,
                    in_channels=in_,
                    out_channels=out_,
                    kernel_size=kernel_size_,
                    stride=1,
                    padding_mode="circular",
                    groups=n_groups,
                    bias = False if norm_config is not None else True
                )
            )
            out_ *= len(dilations)

            if norm_config is not None:
                self.layers.append(
                    LayerNormalization(
                        **dict(norm_config), out_channel=out_, out_size=dim
                    )
                )
            self.layers.append(Activation(activation_specifier=activation_specifier))



        self.residual = residual
        if residual:
            if n_channels[-1] != in_channels:
                raise ValueError(
                    f"In channels {in_channels} do not match out channels {n_channels[-1]}, so residual construction not possible"
                )

        self.concat_input = concat_input
        self._out_channels = n_channels[-1] * len(dilations)
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


class EncoderBlockConfig(BaseModel):

    n_channels: Union[List[int], int]
    residual: bool = False
    activation_specifier: str = "mish"
    norm_configs: Union[List[Union[LayerNormalizationConfig, None]], None] = None
    kernel_sizes: Union[List[int], None] = None
    concat_input: bool = False
    n_groups: int = 1
    dilations: List[int] = [1]

    @validator("n_channels", pre=True)
    @classmethod
    def to_list(cls, v):
        if isinstance(v, int):
            return [v]
        else:
            return v


@ENCODER_REGISTRY.register("cnn")
class CNNEncoder(nn.Module):
    def __init__(
        self,
        conditioner_mask,
        transformed_mask,
        in_channels,
        block_configs: List[EncoderBlockConfig],
        pooling_configs: List[Union[PoolingConfig, None]],
        coord_layer_specifier: Union[bool, str, None] = None,
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


class CNNEncoderConfig(BaseModel):

    block_configs: List[EncoderBlockConfig]
    pooling_configs: List[Union[PoolingConfig, None]] = None


class Encoder(nn.Sequential):
    def __init__(
        self,
        encoder_type,
        conditioner_mask,
        transformed_mask,
        in_channels,
        coord_layer_specifier,
        specific_encoder_config,
    ) -> None:

        self.encoder_parts = []

        if coord_layer_specifier is not None:
            coord_layer = CoordLayer(
                coord_layer_specifier,
                conditioner_mask=conditioner_mask,
                transformed_mask=transformed_mask,
            )
            self.encoder_parts.append(coord_layer)
            in_channels += coord_layer.n_added_channels

        self.encoder_parts.append(
            ENCODER_REGISTRY[encoder_type](
                **dict(specific_encoder_config),
                conditioner_mask=conditioner_mask,
                transformed_mask=transformed_mask,
                in_channels=in_channels,
            )
        )

        super().__init__(*self.encoder_parts)


class EncoderConfig(BaseModel):

    encoder_type: str
    coord_layer_specifier: Optional[
        Literal["rel_position", "abs_position", "rel_position+abs_position"]
    ]=None

    specific_encoder_config: Union[CNNEncoderConfig, DeepLabHeadConfig]
