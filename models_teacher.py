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

from math import ceil

import torch
import torch.nn as nn

from vit import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,intermediate=18,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, pretrain = False, 
                 distill_layer = 'all', distill_dyt = 'all'):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.intermediate=intermediate
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pretrain = pretrain
        self.distill_dyt = distill_dyt
        self.distill_layer = distill_layer
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def match(self, x):
        if x == '6':
            return 0
        if x == '8':
            return 1
        if x == '9':
            return 2
        if x == '11':
            return 3
        if x == '12':
            return 4   
        
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # 改qk, vv为cls
        cls_ls_1 = []
        cls_ls_2 = []
        count = 0

        for blk in self.blocks:
            count += 1
            if self.pretrain and ((self.distill_layer == 'all' and count in [12,15,18,21,24]) 
                                  or (self.distill_layer != 'all' and int(self.distill_layer) == ceil(count/2))):
                # 对teacher的相应层蒸馏，具体层数参考Tiny论文
                x, cls_1, cls_2 = blk(x, return_cls_token=True)
                cls_ls_1.append(cls_1)
                cls_ls_2.append(cls_2)
            else:
                x = blk(x)

        # distill_dyt表示用第几个dyt，distill_layer表示用第几层block
        if self.pretrain:
            if self.distill_dyt == 'before':
                if self.distill_layer != 'all':
                    return cls_ls_1[0]
                else:
                    return torch.stack(cls_ls_1)
            elif self.distill_dyt == 'after':
                if self.distill_layer != 'all':
                    return cls_ls_2[0]
                else:
                    return torch.stack(cls_ls_2)
            elif self.distill_dyt == 'all':
                if self.distill_layer != 'all':
                    return torch.stack([cls_ls_1[0], cls_ls_2[0]])
                else:
                    return torch.stack([torch.stack(cls_ls_1), torch.stack(cls_ls_2)])
        else:
            return x
        
    def forward(self, imgs):
        return self.forward_encoder(imgs)


def mae_vit_small(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,intermediate=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,intermediate=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,intermediate=18,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

