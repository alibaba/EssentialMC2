# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from essmc2.models.registry import BACKBONES, STEMS, BRICKS
from .bricks.non_local import NonLocal
from .init_helper import _init_convnet_weights

_n_conv_resnet = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


class Base3DBlock(nn.Module):
    def __init__(self,
                 branch_name,
                 dim_in,
                 num_filters,
                 kernel_size,
                 downsampling,
                 downsampling_temporal,
                 expansion_ratio,
                 branch_style="simple_block",
                 branch_cfg=None,
                 bn_params=None,
                 visual_cfg=None):
        super(Base3DBlock, self).__init__()

        if dim_in != num_filters or downsampling:
            if downsampling:
                if downsampling_temporal:
                    _stride = (2, 2, 2)
                else:
                    _stride = (1, 2, 2)
            else:
                _stride = (1, 1, 1)
            self.short_cut = nn.Conv3d(
                dim_in,
                num_filters,
                kernel_size=(1, 1, 1),
                stride=_stride,
                padding=0,
                bias=False
            )
            self.short_cut_bn = nn.BatchNorm3d(
                num_filters, **(bn_params or {})
            )
        self.conv_branch = BRICKS.get(branch_name)(dim_in,
                                                   num_filters,
                                                   kernel_size,
                                                   downsampling,
                                                   downsampling_temporal,
                                                   expansion_ratio,
                                                   branch_style=branch_style,
                                                   bn_params=bn_params,
                                                   **(branch_cfg or dict()),
                                                   **(visual_cfg or dict()))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        short_cut = x
        if hasattr(self, "short_cut"):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        x = self.relu(short_cut + self.conv_branch(x))
        return x

    def set_stage_block(self, stage_id, block_id):
        if hasattr(self.conv_branch, "set_stage_block_id"):
            self.conv_branch.set_stage_block_id(stage_id, block_id)


class Base3DResStage(nn.Module):
    """
    ResNet Stage containing several blocks.
    """

    def __init__(
            self,
            num_blocks,
            branch_name,
            dim_in,
            num_filters,
            kernel_size,
            downsampling,
            downsampling_temporal,
            expansion_ratio,
            branch_style="simple_block",
            branch_cfg=None,
            bn_params=None,
            non_local=False,
            non_local_cfg=None,
            visual_cfg=None
    ):
        super(Base3DResStage, self).__init__()
        self.num_blocks = num_blocks

        res_block = Base3DBlock(
            branch_name,
            dim_in,
            num_filters,
            kernel_size,
            downsampling,
            downsampling_temporal,
            expansion_ratio,
            branch_style=branch_style,
            branch_cfg=branch_cfg,
            bn_params=bn_params,
        )
        self.add_module("res_{}".format(1), res_block)
        for i in range(self.num_blocks - 1):
            dim_in = num_filters
            downsampling = False
            res_block = Base3DBlock(
                branch_name,
                dim_in,
                num_filters,
                kernel_size,
                downsampling,
                downsampling_temporal,
                expansion_ratio,
                branch_style=branch_style,
                branch_cfg=branch_cfg,
                bn_params=bn_params,
                visual_cfg=visual_cfg
            )
            self.add_module("res_{}".format(i + 2), res_block)
        if non_local:
            non_local = NonLocal(dim_in, num_filters,
                                 bn_params=bn_params,
                                 **(non_local_cfg or dict()),
                                 **(visual_cfg or dict()))
            self.add_module("nonlocal", non_local)

    def forward(self, x):
        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i + 1))
            x = res_block(x)

        # performs non-local operations if specified.
        if hasattr(self, "nonlocal"):
            non_local = getattr(self, "nonlocal")
            x = non_local(x)
        return x

    def set_stage_id(self, stage_id):
        for i in range(self.num_blocks):
            res_block = getattr(self, "res_{}".format(i + 1))
            res_block.set_stage_block_id(stage_id, i)
        if hasattr(self, "nonlocal"):
            non_local = getattr(self, "nonlocal")
            non_local.set_stage_block_id(stage_id, self.num_blocks)


@BACKBONES.register_class()
class ResNet3D(nn.Module):
    def __init__(self,
                 depth,
                 num_input_channels=3,
                 num_filters=(64, 64, 128, 256, 256),
                 kernel_size=((1, 7, 7), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)),
                 downsampling=(True, False, True, True, True),
                 downsampling_temporal=(False, False, False, True, True),
                 expansion_ratio=2,
                 stem_name="DownSampleStem",
                 branch_name="R2D3DBranch",
                 branch_cfg=None,
                 non_local=(False, False, False, False, False),
                 non_local_cfg=None,
                 bn_params=None,
                 init_cfg=None,
                 visual_cfg=None
                 ):
        super(ResNet3D, self).__init__()

        if bn_params is None:
            bn_params = dict(eps=1e-3, momentum=0.1)

        # Build stem

        stem = dict(
            type="DownSampleStem" if stem_name is None else stem_name,
            dim_in=num_input_channels,
            num_filters=num_filters[0],
            kernel_size=kernel_size[0],
            downsampling=downsampling[0],
            downsampling_temporal=downsampling_temporal[0],
            bn_params=bn_params,
            **visual_cfg or {}
        )
        self.conv1 = STEMS.build(stem)
        self.conv1.set_stage_block_id(0, 0)

        # ------------------- Main arch -------------------
        branch_style = "simple_block" if depth <= 34 else "bottleneck"
        blocks_list = _n_conv_resnet[depth]

        for stage_id, num_blocks in enumerate(blocks_list):
            stage_id = stage_id + 1
            conv = Base3DResStage(
                num_blocks,
                branch_name,
                num_filters[stage_id - 1],
                num_filters[stage_id],
                kernel_size[stage_id],
                downsampling[stage_id],
                downsampling_temporal[stage_id],
                expansion_ratio,
                branch_style=branch_style,
                branch_cfg=branch_cfg,
                bn_params=bn_params,
                non_local=non_local[stage_id],
                non_local_cfg=non_local_cfg,
                visual_cfg=visual_cfg
            )
            setattr(self, f"conv{stage_id + 1}", conv)

        # perform initialization
        init_cfg = init_cfg or dict()
        if init_cfg.get("name") == "kaiming":
            _init_convnet_weights(self)

    def forward(self, video):
        x = self.conv1(video)
        for i in range(2, 6):
            x = getattr(self, f"conv{i}")(x)
        return x


@BACKBONES.register_function("ResNetR2D3D")
def get_resnet_r2d3d(depth=18, **kwargs):
    return BACKBONES.build(dict(
        type="ResNet3D",
        depth=depth,
        **kwargs
    ))


@BACKBONES.register_function("ResNet2Plus1d")
def get_resnet3d_2plus1d(depth=10, **kwargs):
    return BACKBONES.build(dict(
        type="ResNet3D",
        depth=depth,
        num_filters=(64, 64, 128, 256, 512),
        kernel_size=((3, 7, 7), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        downsampling=(True, False, True, True, True),
        downsampling_temporal=(False, False, True, True, True),
        expansion_ratio=2,
        stem_name="R2Plus1DStem",
        branch_name="R2Plus1DBranch",
        **kwargs
    ))


@BACKBONES.register_function("ResNet3D_CSN")
def get_resnet3d_csn(depth=152, **kwargs):
    return BACKBONES.build(dict(
        type="ResNet3D",
        depth=depth,
        num_filters=(64, 256, 512, 1024, 2048),
        kernel_size=((3, 7, 7), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        downsampling=(True, False, True, True, True),
        downsampling_temporal=(False, False, True, True, True),
        expansion_ratio=4,
        stem_name="DownSampleStem",
        branch_name="CSNBranch",
        **kwargs
    ))


@BACKBONES.register_function("ResNet3D_TAda")
def get_resnet3d_TAda(depth=50, **kwargs):
    return BACKBONES.build(dict(
        type="ResNet3D",
        depth=depth,
        num_filters=(64, 256, 512, 1024, 2048),
        kernel_size=((1, 7, 7), (1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
        downsampling=(True, True, True, True, True),
        downsampling_temporal=(False, False, False, False, False),
        expansion_ratio=4,
        stem_name="Base2DStem",
        branch_name="TAdaConvBlockAvgPool",
        branch_cfg=dict(
            route_func_k=(3, 3),
            route_func_r=4,
            pool_k=(3, 1, 1)
        ),
        init_cfg=dict(name="kaiming"),
        **kwargs
    ))
