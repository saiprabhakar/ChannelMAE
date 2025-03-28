# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block, HybridEmbed
from util.pos_embed import get_2d_sincos_pos_embed
from collections import OrderedDict
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_


class ChannelVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, global_pool=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            TODO add global pool
        """
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=1, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.channel_embed = nn.Embedding(in_chans, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, nc, H, W = x.shape  # torch.Size([2, 3, 224, 224])
        # x: [B, nc, H, W]
        # self.pos_embed: torch.Size([1, 197, 128])
        # self.cls_token: torch.Size([1, 1, 128])

        # embed the patches by merging the channels and batch dimensions
        # x: [B, nc, H, W] -> [B, L, D] ; L = H*W*nc/(p*p)
        x = x.reshape(-1, 1, H, W)  # torch.Size([6, 1, 224, 224])
        x = self.patch_embed(x)  # [B*nc, 1, L, D] # torch.Size([6, 196, 128])
        x = x.reshape(
            B, nc, -1, x.shape[-1]
        )  # [B, nc, L, D] # torch.Size([2, 3, 196, 128])
        L = x.shape[2]  # 196

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :]  # torch.Size([1, 196, 128])
        cls_pos_embed = self.pos_embed[:, :1, :]  # torch.Size([1, 1, 128])
        pos_embed = pos_embed.unsqueeze(1)  # [B, L, D] # torch.Size([1, 1, 196, 128])
        x = x + pos_embed  # [B, ch, L, D] # torch.Size([2, 3, 196, 128])

        # add channel embed, not for cls token
        channels = torch.arange(nc, device=x.device).long()  # [0,1,2]
        channel_embed = self.channel_embed(channels)  # [nc, D] # torch.Size([3, 128])
        channel_embed = channel_embed.unsqueeze(0).unsqueeze(
            2
        )  # [1, nc, 1, D] # torch.Size([1, 3, 1, 128])
        x = x + channel_embed  # [B, nc, L, D] # torch.Size([2, 3, 196, 128])

        x = x.reshape(B, L * nc, -1)  # [B*nc, L, D] # torch.Size([2, 588, 128])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # torch.Size([1, 1, 128])
        cls_tokens = cls_token.expand(B, -1, -1)  # torch.Size([2, 1, 128])
        x = torch.cat((cls_tokens, x), dim=1)  # x: [2, 148, 128]

        # apply Transformer blocks
        for blk in self.blocks:  # x: [2, 148, 128]
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome # [2, 128]

    def forward(self, x):
        x = self.forward_features(x) # torch.Size([4, 3, 224, 224]) -> torch.Size([4, 128])
        x = self.head(x)
        return x


def channelvit_tiny_patch16(**kwargs):
    model = ChannelVisionTransformer(
        patch_size=16, embed_dim=32, depth=1, num_heads=4, mlp_ratio=2, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def channelvit_base_patch16(**kwargs):
    model = ChannelVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def channelvit_large_patch16(**kwargs):
    model = ChannelVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def channelvit_huge_patch14(**kwargs):
    model = ChannelVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model