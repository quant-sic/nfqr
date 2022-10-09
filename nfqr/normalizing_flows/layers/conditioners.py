from typing import Literal, Union

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import Module

from nfqr.normalizing_flows.nets import (
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    DecoderConfig,
    EncoderConfig,
)
from nfqr.normalizing_flows.nets.decoder import CNNDecoder
from nfqr.registry import StrRegistry
from einops.layers.torch import Rearrange

ENCODER_CONDITIONER_REGISTRY = StrRegistry("encoders")
DECODER_CONDITIONER_REGISTRY = StrRegistry("decoders")


@ENCODER_CONDITIONER_REGISTRY.register("u1")
class U1Encoder(Module):
    def __init__(self, in_channels, encoder_config, conditioner_mask, transformed_mask):
        super(U1Encoder, self).__init__()

        dim_in = conditioner_mask.sum().item()

        if encoder_config is None:
            self.net = lambda x: x
            self._dim_out = dim_in
            self._out_channels = 2 * in_channels
        else:
            self.net = ENCODER_REGISTRY[encoder_config.encoder_type](
                in_size=dim_in,
                in_channels=2 * in_channels,
                **dict(encoder_config.specific_encoder_config),
                conditioner_mask=conditioner_mask,
                transformed_mask=transformed_mask,
            )
            self._dim_out = self.net.dim_out
            self._out_channels = self.net.out_channels

    @property
    def dim_out(self):
        return self._dim_out

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, z):

        z_u1 = torch.stack(
            [
                torch.cos(z),
                torch.sin(z),
            ],
            1,
        )

        out = self.net(z_u1)

        return out


@DECODER_CONDITIONER_REGISTRY.register("u1")
class U1Decoder(Module):
    def __init__(
        self,
        dim_in,
        in_channels,
        expressivity,
        num_splits,
        num_extra_single_pars,
        decoder_config,
        transformed_mask,
        num_nets: int = 1,
    ):
        super(U1Decoder, self).__init__()

        dim_out = transformed_mask.sum().item()

        if num_nets==0 or (not dim_out % num_nets == 0):
            raise ValueError(
                f"dim_out({dim_out}) must be divisible by num_nets ({num_nets}) supported"
            )
        dim_out_per_net = int(dim_out / num_nets) if num_nets!=0 else dim_out

        self.nets = nn.ModuleList()

        self.out_channels = expressivity * num_splits + num_extra_single_pars

        if not decoder_config is None:
            for _ in range(num_nets):

                if isinstance(DECODER_REGISTRY[decoder_config.decoder_type],CNNDecoder):
                    if not dim_in==dim_out_per_net:
                        raise ValueError("For CNN decoder insize must be equal to out size")
                        
                self.nets.append(
                    DECODER_REGISTRY[decoder_config.decoder_type](
                        in_size=dim_in,
                        in_channels=in_channels,
                        out_size=dim_out_per_net,
                        out_channels=self.out_channels,
                        **dict(decoder_config.specific_decoder_config),
                    )
                )

        else:
            if not in_channels == self.out_channels or not dim_in==dim_out_per_net:
                raise ValueError(f"There needs to be a net if in_channels ({in_channels}) != out_channels ({self.out_channels}), dim_in ({dim_in}) != dim_out ({dim_out_per_net})") 
            else:
                self.nets.append(nn.Sequential(Rearrange("a b c -> a c b")))

        self.expressivity = expressivity
        self.num_splits = num_splits
        self.num_extra_single_pars = num_extra_single_pars

    def forward(self, z):

        z = torch.cat([net(z) for net in self.nets], dim=1)
        h_pars = torch.split(z, [self.expressivity]*self.num_splits + [1]*self.num_extra_single_pars, dim=-1)

        return h_pars


class ConditionerChain(Module):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
        share_encoder: bool,
        share_decoder: bool,
        num_pars: int,
        expressivity: int,
        layer_splits,
        num_extra_single_pars:int=0,
        num_enocders: int = 1,
        num_decoders: int = 1,
        domain="u1",
    ) -> None:
        super().__init__()

        self.domain = domain
        self.layer_splits = layer_splits

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.num_encoders = num_enocders
        self.num_decoders = num_decoders

        self.num_pars = num_pars
        self.num_extra_single_pars=num_extra_single_pars
        self.expressivity = expressivity

        self.share_encoder = share_encoder
        if share_encoder:
            if not layer_splits.all_conditioners_equal_in:
                raise RuntimeError(
                    "share conditioners not possible since in and output dimensions do not match"
                )

        self.share_decoder = share_decoder
        if share_decoder:
            if not layer_splits.all_conditioners_equal_out:
                raise RuntimeError(
                    "share conditioners not possible since in and output dimensions do not match"
                )

    def make_encoder(self, conditioner_mask, transformed_mask):
        return ENCODER_CONDITIONER_REGISTRY[self.domain](
            in_channels=1,
            encoder_config=self.encoder_config,
            conditioner_mask=conditioner_mask,
            transformed_mask=transformed_mask,
        )

    def make_decoder(self, dim_in, in_channels, transformed_mask):
        return DECODER_CONDITIONER_REGISTRY[self.domain](
            dim_in=dim_in,
            in_channels=in_channels,
            expressivity=self.expressivity,
            num_splits=self.num_pars,
            num_extra_single_pars=self.num_extra_single_pars,
            decoder_config=self.decoder_config,
            transformed_mask=transformed_mask,
            num_nets=self.num_decoders,
        )

    def __iter__(self):

        conditioner_mask, transformed_mask = self.layer_splits.__iter__().__next__()
        shared_encoder = self.make_encoder(
            conditioner_mask=conditioner_mask, transformed_mask=transformed_mask
        )

        shared_decoder = self.make_decoder(
            dim_in=shared_encoder.dim_out,
            in_channels=shared_encoder.out_channels,
            transformed_mask=transformed_mask,
        )

        for conditioner_mask, transformed_mask in self.layer_splits:

            if self.share_encoder:
                encoder = shared_encoder
            else:
                encoder = self.make_encoder(
                    conditioner_mask=conditioner_mask, transformed_mask=transformed_mask
                )

            if self.share_decoder:
                decoder = shared_decoder
            else:
                decoder = self.make_decoder(
                    dim_in=encoder.dim_out,
                    in_channels=encoder.out_channels,
                    transformed_mask=transformed_mask,
                )

            yield nn.Sequential(encoder, decoder)

        if not self.share_decoder:
            del shared_decoder
        if not self.share_encoder:
            del shared_encoder


class ConditionerChainConfig(BaseModel):

    domain: Literal["u1"] = "u1"
    encoder_config: Union[EncoderConfig, None]
    decoder_config: Union[DecoderConfig,None]
    share_encoder: bool
    share_decoder: bool
    expressivity: int
    num_enocders: int = 1
    num_decoders: int = 1
    # validators ..
