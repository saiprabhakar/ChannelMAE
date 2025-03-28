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

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderChaViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,  # TODO implement this
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # ChaMAE encoder specifics
        self.in_chans = in_chans
        self.patch_size = patch_size
        # we have same filter for all channels
        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # we have channel specifc embedding in addition to positional embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        self.decoder_channel_embed = nn.Embedding(in_chans, decoder_embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # channels are flattened to 1D so the dim for each pactch is patch_size**2 and not patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L*C, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]  # 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p  # 14
        x = imgs.reshape(
            shape=(imgs.shape[0], self.in_chans, h, p, w, p)
        )  # B, nc, h, p, w, p # torch.Size([2, 3, 14, 16, 14, 16])
        x = torch.einsum(
            "nchpwq->nchwpq", x
        )  # B, h, w, nc, p, p # torch.Size([2, 3, 14, 14, 16, 16])
        x = x.reshape(
            shape=(imgs.shape[0], self.in_chans * h * w, p**2)
        )  # torch.Size([2, 588, 256])
        x_un = self.unpatchify(x)
        assert torch.allclose(imgs, x_un)
        return x

    def unpatchify(self, x):
        """
        x: (N, L*C, patch_size**2)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int((x.shape[1] / self.in_chans) ** 0.5)
        assert h * w == x.shape[1] // self.in_chans

        x = x.reshape(
            shape=(x.shape[0], self.in_chans, h, w, p, p)
        )  # [N, C, h, w, p, p]
        x = torch.einsum("nchwpq->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
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

        # implement random masking by laying out the channels and patches in the same dimension
        x = x.reshape(B, L * nc, -1)  # [B*nc, L, D] # torch.Size([2, 588, 128])
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio
        )  # x: [2, 147, 128] # mask [2, 588] # ids_restore [2, 588]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # torch.Size([1, 1, 128])
        cls_tokens = cls_token.expand(B, -1, -1)  # torch.Size([2, 1, 128])
        x = torch.cat((cls_tokens, x), dim=1)  # x: [2, 148, 128]

        # apply Transformer blocks
        for blk in self.blocks:  # x: [2, 148, 128]
            x = blk(x)
        x = self.norm(x)  # x: [2, 148, 128]

        return (
            x,
            mask,
            ids_restore,
        )  # x: [2, 148, 128], mask: [2, 588], ids_restore: [2, 588]

    def forward_decoder(self, x, ids_restore):
        # x: [2, 148, 128]
        # ids_restore: [2, 588]
        # self.mask_token: torch.Size([1, 1, 64])
        # self.docoder_pos_embed: torch.Size([1, 197, 64])

        # embed the unmasked tokens
        x = self.decoder_embed(x)  # [2, 148, 64]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )  # torch.Size([2, 441, 64])
        x_ = torch.cat(
            [x[:, 1:, :], mask_tokens], dim=1
        )  # no cls token # torch.Size([2, 588, 64])

        # unshuffle the masked and unmasked tokens
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # torch.Size([2, 588, 64])

        # approach 2
        # first add the channel embed then the positional embed

        # add channel embed to all but cls token
        channels = torch.arange(self.in_chans, device=x.device).long()  # channels
        dec_channel_embed = self.decoder_channel_embed(
            channels
        )  # [nc, D] # torch.Size([3, 64])
        dec_channel_embed = dec_channel_embed.unsqueeze(0).unsqueeze(
            2
        )  # [1, nc, 1, D] # torch.Size([1, 3, 1, 64])
        x_2 = x_.reshape(
            x.shape[0], self.in_chans, -1, x.shape[-1]
        )  # torch.Size([2, 3, 196, 64])
        x_2 = x_2 + dec_channel_embed  # [B, nc, L, D] # torch.Size([2, 3, 196, 64])

        # add positional embed to all but cls token
        dec_pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(
            1
        )  # [1, L, D] -> [1, 1, L, D] # torch.Size([1, 1, 196, 64])
        x_2_w_pos_embed = x_2 + dec_pos_embed  # torch.Size([2, 3, 196, 64])
        x_2_w_pos_embed = x_2_w_pos_embed.reshape(
            x.shape[0], -1, x.shape[-1]
        )  # [B, nc*L, D] # torch.Size([2, 588, 64])

        x_cls = x[:, :1, :] + self.decoder_pos_embed[:, :1, :]  # torch.Size([2, 1, 64])
        x2 = torch.cat([x_cls, x_2_w_pos_embed], dim=1)

        # assert torch.allclose(x1,x2, rtol=1e-06, atol=1e-06)

        x = x2

        # apply Transformer blocks
        for blk in self.decoder_blocks:  # torch.Size([2, 589, 64])
            x = blk(x)
        x = self.decoder_norm(x)  # torch.Size([2, 589, 64])

        # predictor projection
        x = self.decoder_pred(x)  # torch.Size([2, 589, 256])

        # remove cls token
        x = x[:, 1:, :]  # B, L*c, p*p # torch.Size([2, 588, 256])
        return x  # torch.Size([2, 588, 256])

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L*C, p*p]
        mask: [N, L*C], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio
        )  # x: [2, 148, 128], mask: [2, 588], ids_restore: [2, 588]

        pred = self.forward_decoder(
            latent, ids_restore
        )  # [N, L, p*p*C] # torch.Size([2, 588, 256])
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask  # loss: scalar, pred: [2, 588, 256], mask: [2, 588]


def chmae_vit_tiny_testing_patch16_dec32d1b(**kwargs):
    model = MaskedAutoencoderChaViT(
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def chmae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderChaViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def chmae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderChaViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def chmae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderChaViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

chmae_vit_tiny_testing = chmae_vit_tiny_testing_patch16_dec32d1b
chmae_vit_base_patch16 = chmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
chmae_vit_large_patch16 = chmae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
chmae_vit_huge_patch14 = chmae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks