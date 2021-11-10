# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional
from einops import rearrange

from essmc2.models.registry import BACKBONES, STEMS, BRICKS
from .init_helper import trunc_normal_, _init_transformer_weights


@BACKBONES.register_class()
class Transformer(nn.Module):
    def __init__(self,
                 num_input_channels=3,
                 num_frames=8,
                 image_size=224,
                 num_features=768,
                 patch_size=16,
                 depth=12,
                 drop_path=0.1,
                 tubelet_size=2,
                 stem_name="TubeletEmbeddingStem",
                 branch_name="BaseTransformerLayer",
                 branch_cfg=None
                 ):
        super(Transformer, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divided by patch size.'

        self.path_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        assert stem_name in ('PatchEmbedStem', 'TubeletEmbeddingStem')
        stem_cfg = dict(
            type=stem_name,
            image_size=image_size,
            num_input_channels=num_input_channels,
            num_frames=num_frames,
            patch_size=patch_size,
            dim=num_features,
        )
        if stem_name == "TubeletEmbeddingStem":
            stem_cfg["tubelet_size"] = tubelet_size

        self.stem = STEMS.build(stem_cfg)

        self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))

        assert branch_name in ('BaseTransformerLayer', 'TimesformerLayer')
        if branch_name == "BaseTransformerLayer":
            branch_cfg = dict(
                type=branch_name,
                dim=num_features,
                **(branch_cfg or dict())
            )
        elif branch_name == "TimesformerLayer":
            branch_cfg = dict(
                type=branch_name,
                num_patches=(image_size // patch_size) ** 2,
                num_frames=num_frames,
                dim=num_features,
                **(branch_cfg or dict())
            )
        else:
            raise RuntimeError(f"Not supported branch_name {branch_name}")
        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        layers = []
        for i in range(depth):
            branch_cfg["drop_path_prob"] = dpr[i]
            layers.append(BRICKS.build(branch_cfg))
        self.layers = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, video):
        x = video
        x = self.stem(x)

        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)

        return x[:, 0]


@BACKBONES.register_class()
class FactorizedTransformer(nn.Module):
    def __init__(self,
                 num_input_channels=3,
                 num_frames=8,
                 image_size=224,
                 num_features=768,
                 patch_size=16,
                 depth=12,
                 depth_temp=4,
                 drop_path=0.1,
                 tubelet_size=2,
                 stem_name="TubeletEmbeddingStem",
                 branch_name="BaseTransformerLayer",
                 branch_cfg=None
                 ):
        super(FactorizedTransformer, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divided by patch size.'

        self.patch_size = patch_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        assert stem_name in ('PatchEmbedStem', 'TubeletEmbeddingStem')
        stem_cfg = dict(
            type=stem_name,
            image_size=image_size,
            num_input_channels=num_input_channels,
            num_frames=num_frames,
            patch_size=patch_size,
            dim=num_features,
        )
        if stem_name == "TubeletEmbeddingStem":
            stem_cfg["tubelet_size"] = tubelet_size

        self.stem = STEMS.build(stem_cfg)

        self.pos_embd = nn.Parameter(torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd = nn.Parameter(torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        assert branch_name in ('BaseTransformerLayer', 'TimesformerLayer')
        if branch_name == "BaseTransformerLayer":
            branch_cfg = dict(
                type=branch_name,
                dim=num_features,
                **(branch_cfg or dict())
            )
        elif branch_name == "TimesformerLayer":
            branch_cfg = dict(
                type=branch_name,
                num_patches=(image_size // patch_size) ** 2,
                num_frames=num_frames,
                dim=num_features,
                **(branch_cfg or dict())
            )
        else:
            raise RuntimeError(f"Not supported branch_name {branch_name}")
        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth + depth_temp)]  # stochastic depth decay rule

        layers = []
        for i in range(depth):
            branch_cfg["drop_path_prob"] = dpr[i]
            layers.append(BRICKS.build(branch_cfg))
        self.layers = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # construct temporal transformer layers
        layers_temporal = []
        for i in range(depth_temp):
            branch_cfg["drop_path_prob"] = dpr[i + depth]
            layers_temporal.append(BRICKS.build(branch_cfg))
        self.layers_temporal = nn.Sequential(*layers_temporal)

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, video):
        x = video
        h, w = x.shape[-2:]
        actual_num_patches_per_frame = (h // self.patch_size) * (w // self.patch_size)
        x = self.stem(x)

        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training
            x = rearrange(x, "b (t n) c -> (b t) n c", n=actual_num_patches_per_frame)
        else:
            x = rearrange(x, "b (t n) c -> (b t) n c", n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self, "new_pos_embd") or self.new_pos_embd.shape[1] != (
                    actual_num_pathces_per_side ** 2 + 1):
                cls_pos_embd = self.pos_embd[:, 0, :].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side, num_patches_per_side, -1).permute(0, 3, 1, 2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd, size=(actual_num_pathces_per_side, actual_num_pathces_per_side), mode="bilinear"
                ).permute(0, 2, 3, 1).reshape(1, actual_num_pathces_per_side ** 2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x, "(b t) c -> b t c", t=self.num_patches // self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd

        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]


@BACKBONES.register_function("ViVit")
def get_transformer_vivit(image_size=112, num_frames=16):
    return BACKBONES.build(dict(
        type="Transformer",
        num_frames=num_frames,
        image_size=image_size,
        num_features=768,
        patch_size=16,
        depth=12,
        drop_path=0.1,
        tubelet_size=2,
        stem_name="TubeletEmbeddingStem",
        branch_name="BaseTransformerLayer",
        branch_cfg=dict(
            num_heads=12,
            attn_dropout=0.0,
            ff_dropout=0.0,
            mlp_mult=4
        )
    ))


@BACKBONES.register_function("ViVit_Fac_Enc")
def get_transformer_fac_enc(image_size=112, num_frames=16, **kwargs):
    cfg = dict(
        type="FactorizedTransformer",
        num_frames=num_frames,
        image_size=image_size,
        num_features=768,
        patch_size=16,
        depth=12,
        depth_temp=4,
        drop_path=0.1,
        tubelet_size=2,
        stem_name="TubeletEmbeddingStem",
        branch_name="BaseTransformerLayer",
        branch_cfg=dict(
            num_heads=12,
            attn_dropout=0.0,
            ff_dropout=0.0,
            mlp_mult=4
        )
    )
    if kwargs:
        cfg.update(kwargs)
    return BACKBONES.build(cfg)


@BACKBONES.register_function("Timesformer")
def get_timesformer(image_size=112, num_frames=16):
    return BACKBONES.build(dict(
        type="Transformer",
        num_frames=num_frames,
        image_size=image_size,
        num_features=768,
        patch_size=16,
        depth=12,
        drop_path=0.0,
        tubelet_size=1,
        stem_name="PatchEmbedStem",
        branch_name="TimesformerLayer",
        branch_cfg=dict(
            num_heads=12,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
    ))
