"""
https://github.com/danfenghong/Codes_SpectralFormer/blob/0a7ee007e9421418e52c23dcee348be6302ec88c/vit_pytorch.py
"""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout))
                        ),
                    ]
                )
            )

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(
                nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)
            )

    def forward(self, x, mask=None):
        if self.mode == "ViT":
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == "CAF":
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat(
                            [x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3
                        )
                    ).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x


class ViT(pl.LightningModule):
    def __init__(
        self,
        number_of_spatial_pixels,
        band_grouping_size,
        spectral_dim,
        transformer_internal_dim,
        number_of_transformer_encoders,
        number_of_attention_heads,
        mlp_dim,
        pool="cls",
        channels=1,
        dim_head=16,
        dropout=0.0,
        emb_dropout=0.0,
        mode="ViT",
    ):
        super().__init__()

        self.transformer_internal_dim = transformer_internal_dim

        patch_dim = number_of_spatial_pixels * band_grouping_size

        self.pos_embedding = nn.Parameter(
            torch.randn(1, spectral_dim + 1, transformer_internal_dim)
        )
        self.patch_to_embedding = nn.Linear(patch_dim, transformer_internal_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_internal_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            transformer_internal_dim,
            number_of_transformer_encoders,
            number_of_attention_heads,
            dim_head,
            mlp_dim,
            dropout,
            spectral_dim,
            mode,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.gse_pb = lambda x: torch.stack(
            [
                torch.roll(x, dims=-1, shifts=int(np.floor(band_grouping_size / 2)) - i)
                for i in range(band_grouping_size)
            ],
            dim=-1,
        )

    def forward(self, x, mask=None):

        if x.ndim > 2:
            x = x.squeeze()

        x = self.gse_pb(x)
        # x = rearrange(x, 'b s p -> s b')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]

        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # # MLP classification layer
        # x = self.mlp_head(x)

        return x

    # def reset_mlp_head(self, number_of_classes):

    #     device = self.mlp_head.parameters().__next__().device
    #     self.mlp_head = nn.Sequential(
    #         nn.LayerNorm(self.transformer_internal_dim),
    #         nn.Linear(self.transformer_internal_dim,
    #                   number_of_classes)).to(device=device)
    #     self.mlp_head.requires_grad = True

    # def set_encoder_requires_grad(self, requires_grad=True):

    #     for name, parameter in self.named_parameters():

    #         if not name.split(".")[0] == "mlp_head":
    #             parameter.requires_grad = requires_grad


class MLP_Head(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.mlp_head(x)
        x = self.log_softmax(x)
        return x
