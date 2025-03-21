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
from vit import PatchEmbed, Block, Dyt_Block
from util.pos_embed import get_2d_sincos_pos_embed


class TinyMIMViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, depth=24, num_heads=16,last_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.last_heads = last_heads
        self.blocks = nn.ModuleList([
            Dyt_Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Dyt_Block(embed_dim, self.last_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        self.dyt_proj = nn.Linear(embed_dim, 1024)  # 学生投影：768 → 1024
        self.teacher_proj = nn.Linear(1024, 1024)

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
        count=0
        for blk in self.blocks:
            count+=1
            if count==12:
                qk, vv, dyt_x = blk(x, return_relation=True)
                return qk, vv, dyt_x
            else:
                x = blk(x)
        return x

    def forward_kd_loss(self, pred, teacher_out):
        pred = pred.log()
        loss = nn.KLDivLoss(reduction="none")(pred, teacher_out).sum(-1)
        return loss.mean()
    def forward_kd_softmax_loss(self, pred, teacher_out):
        # 由于dyt和norm的输出空间不是softmax，而kl散度要求输入是过softmax的（qk和vv最后都有过softmax）
        student_log_probs = torch.log_softmax(pred, dim=-1)
        teacher_probs = torch.softmax(teacher_out, dim=-1)

        # 添加数值稳定性保护
        teacher_probs = teacher_probs.clamp(min=1e-8)  # 防止除零

        # 计算 KL 散度
        loss = nn.KLDivLoss(reduction="none")(student_log_probs, teacher_probs).sum(-1)
        return loss.mean()

    def forward(self, imgs, teacher_out): # qk和vv最后有过softmax，但dyt和norm没有，计算kl散度之前要过softmax
        qk, vv, dyt_x = self.forward_encoder(imgs)
        qk_loss = self.forward_kd_loss(qk, teacher_out[0])
        vv_loss = self.forward_kd_loss(vv, teacher_out[1])

        dyt_proj = self.dyt_proj(dyt_x)  # 维度变为 (4, 197, 1024)
        teacher_proj = self.teacher_proj(teacher_out[2])
        dyt_loss = self.forward_kd_softmax_loss(dyt_proj, teacher_proj) # 加了一个dyt loss，teacher的norm和student的dyt做对比
        return qk_loss, vv_loss, dyt_loss


def tinymim_vit_tiny_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, drop_path=0.1,last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tinymim_vit_small_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,drop_path=0.1,last_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def tinymim_vit_base_patch16(**kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1,last_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


