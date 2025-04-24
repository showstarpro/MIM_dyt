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
from vit import PatchEmbed, Block, Dyt_Block, DyT
from util.pos_embed import get_2d_sincos_pos_embed


class TinyMIMViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, tea_embed_dim=1024, depth=24, num_heads=16, last_heads=12,
                 mlp_ratio=4., norm_layer="dyt", layer=None, num_classes=1000,
                 finetune=False):
        super().__init__()

        # --------------------------------------------------------------------------
        self.finetune = finetune

        if layer is None:
            layer = [12]
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.last_heads = last_heads

        self.norm_type = norm_layer.lower()  # 统一为小写
        # 根据norm类型选择归一化层
        if self.norm_type == "dyt":
            self.norm_layer = DyT(embed_dim, alpha_init_value=0.5)
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path,
                                                qkv_bias=True, qk_scale=None, norm_layer=DyT) for _ in range(depth - 1)
                                        ] + [Block(embed_dim, self.last_heads, mlp_ratio,
                                                qkv_bias=True, qk_scale=None, norm_layer=DyT)])
        elif self.norm_type == "norm":
            self.norm_layer = nn.LayerNorm(embed_dim)
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path,
                                                qkv_bias=True, qk_scale=None) for _ in range(depth - 1)
                                        ] + [Block(embed_dim, self.last_heads, mlp_ratio,
                                                qkv_bias=True, qk_scale=None)])
        else:
            # 为fusion模式选择最终的norm层，这里使用DyT
            self.norm_layer = DyT(embed_dim, alpha_init_value=0.5)
            
            layers = norm_layer.split('_')
            
            block_comb = []
            
            for block in layers:
                if block == "dyt":
                    block_comb.append(Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path,
                                       qkv_bias=True, qk_scale=None, norm_layer=DyT))
                elif block == "norm":
                    block_comb.append(Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path,
                                       qkv_bias=True, qk_scale=None))
                else:
                    raise ValueError(f"Unsupported norm layer: {block}")
            
            # 创建模块列表
            blocks = []
            for i in range(3):
                blocks.extend(block_comb)
            
            self.blocks = nn.ModuleList(blocks)

        # --------------------------------------------------------------------------

        self.initialize_weights()
        # self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 线性层
        # self.dyt_proj = nn.Linear(embed_dim, tea_embed_dim)  # 学生投影：768 → 1024
        # self.teacher_proj = nn.Linear(tea_embed_dim, tea_embed_dim)

        # self.dyt_f_proj = nn.Linear(embed_dim, tea_embed_dim)  # 学生投影：768 → 1024
        # self.teacher_f_proj = nn.Linear(tea_embed_dim, tea_embed_dim)

        # self.x_proj = nn.Linear(embed_dim, tea_embed_dim)  # 输出投影：768 → 1024
        # self.t_proj = nn.Linear(tea_embed_dim, tea_embed_dim)

        # self.clsx_proj = nn.Linear(embed_dim, tea_embed_dim)  # 输出投影：768 → 1024
        # self.clst_proj = nn.Linear(tea_embed_dim, tea_embed_dim)

        self.layer = layer

        # finetune用的分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
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

        # apply Transformer dyt blocks
        count = 0
        qk = []
        vv = []
        norm_x = []
        norm_x_f = []
        cls = []
        for blk in self.blocks:
            count += 1
            if count in self.layer:
                x, qk_temp, vv_temp, norm_temp, norm_temp_f = blk(x, return_relation=True)
                qk.append(qk_temp)
                vv.append(vv_temp)
                norm_x.append(norm_temp)
                norm_x_f.append(norm_temp_f)
                # cls.append(x[:, 0, :])
                cls.append(x)
            else:
                x, _, _, _, _ = blk(x, return_relation=True)
        return x, qk, vv, norm_x, norm_x_f, cls

    def forward_kd_loss(self, pred, teacher_out): # base line里面算qk和vv的用kl，不需要更改
        pred = pred.log()
        loss = nn.KLDivLoss(reduction="none")(pred, teacher_out).sum(-1)
        return loss.mean()

    def forward_L2_loss(self, pred, teacher_out):
        loss = nn.MSELoss()(pred, teacher_out)
        return loss

    def forward(self, imgs, teacher_out=None):  # qk和vv最后有过softmax，但dyt和norm没有，计算kl散度之前要过softmax
        # 由于base和large的注意力投影头数目不一样，所以中间层的qk和vv无法做loss（tinny中把base的最后一层改成和large投影头一样了）
        # 此处先按照原本的做，可以同样更改头数目来做中间的qk和vv loss

        if teacher_out is None:  # 不输入教师的输出就默认微调
            return self.forward_finetune(imgs)  # 微调模式
        else:
            x, qk, vv, dyt_x, dyt_x_f, cls = self.forward_encoder(imgs)

            # 初始化各损失项
            qk_loss = vv_loss = 0.0
            dyt_loss = dyt_f_loss = 0.0
            cls_loss = 0.0

            # 遍历所有指定层
            for i in range(len(self.layer)):
                # 计算 qk 和 vv 的最后一层损失
                # if i == len(self.layer) - 1:
                #     qk_loss += self.forward_kd_loss(qk[i], teacher_out[1][i])
                #     vv_loss += self.forward_kd_loss(vv[i], teacher_out[2][i])

                # 处理 norm(attn) 和 norm(fnn) 的多层损失
                # ---------------------
                # 计算 norm(attn) 的损失
                s_dyt = dyt_x[i]  # 学生输出投影
                dyt_loss += self.forward_L2_loss(s_dyt, teacher_out[3][i])

                # 计算 norm(fnn) 的损失
                s_dyt_f = dyt_x_f[i]
                dyt_f_loss += self.forward_L2_loss(s_dyt_f, teacher_out[4][i])

                # 计算 cls 的损失
                s_cls = cls[i]
                cls_loss += self.forward_L2_loss(s_cls, teacher_out[5][i])

            # 平均损失（按层数）
            num_layers = len(self.layer)
            # qk_loss /= num_layers
            # vv_loss /= num_layers
            dyt_loss /= num_layers
            dyt_f_loss /= num_layers
            cls_loss /= num_layers

            return qk_loss, vv_loss, dyt_loss, dyt_f_loss, cls_loss

    def forward_finetune(self, imgs):
        """微调用前向传播"""
        x, _, _, _, _, _ = self.forward_encoder(imgs)  # 仅用分类输出
        x = self.norm_layer(x[:, 0])  # 取分类令牌
        return self.head(x)  # 输出logits

    def no_weight_decay(self):
        """返回不需要权重衰减的参数名列表"""
        no_decay = {'pos_embed', 'cls_token'}
        return no_decay


def tinymim_vit_tiny_patch16(norm="dyt", **kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=192, depth=12, tea_embed_dim=1024, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=norm, **kwargs)
    return model


def tinymim_vit_small_patch16(norm="dyt", **kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=384, depth=12, tea_embed_dim=1024, num_heads=6, drop_path=0.1, last_heads=12,
        mlp_ratio=4, norm_layer=norm, **kwargs)
    return model


def tinymim_vit_base_patch16(norm="dyt", last_head=16, tea_embed_dim=1024, **kwargs):
    model = TinyMIMViT(
        patch_size=16, embed_dim=768, tea_embed_dim=tea_embed_dim, depth=12, num_heads=12, drop_path=0.1, last_heads=last_head,
        mlp_ratio=4, norm_layer=norm, **kwargs)
    return model


