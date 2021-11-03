# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import math

import torch.nn as nn

from essmc2.models.registry import BRICKS
from .base_branch import BaseBranch


@BRICKS.register_class()
class R2Plus1DBranch(BaseBranch):
    def __init__(self,
                 dim_in,
                 num_filters,
                 kernel_size,
                 downsampling,
                 downsampling_temporal,
                 expansion_ratio,
                 bn_params=None,
                 **kwargs):
        self.dim_in = dim_in
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.downsampling = downsampling
        self.downsampling_temporal = downsampling_temporal
        self.expansion_ratio = expansion_ratio
        self.bn_params = bn_params or {}
        if self.downsampling:
            if self.downsampling_temporal:
                self.stride = (2, 2, 2)
            else:
                self.stride = (1, 2, 2)
        else:
            self.stride = (1, 1, 1)
        super(R2Plus1DBranch, self).__init__(**kwargs)

    def _construct_simple_block(self):
        mid_dim = int(
            math.floor(
                (self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.dim_in * self.num_filters) / \
                (self.kernel_size[1] * self.kernel_size[2] * self.dim_in + self.kernel_size[0] * self.num_filters)))

        self.a1 = nn.Conv3d(
            in_channels=self.dim_in,
            out_channels=mid_dim,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
            bias=False
        )
        self.a1_bn = nn.BatchNorm3d(mid_dim, **self.bn_params)
        self.a1_relu = nn.ReLU(inplace=True)

        self.a2 = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.kernel_size[0] // 2, 0, 0),
            bias=False
        )
        self.a2_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)
        self.a2_relu = nn.ReLU(inplace=True)

        mid_dim = int(
            math.floor((self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[
                2] * self.num_filters * self.num_filters) / \
                       (self.kernel_size[1] * self.kernel_size[2] * self.num_filters + self.kernel_size[
                           0] * self.num_filters)))

        self.b1 = nn.Conv3d(
            in_channels=self.num_filters,
            out_channels=mid_dim,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            stride=(1, 1, 1),
            padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
            bias=False
        )
        self.b1_bn = nn.BatchNorm3d(mid_dim, **self.bn_params)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size[0], 1, 1),
            stride=(1, 1, 1),
            padding=(self.kernel_size[0] // 2, 0, 0),
            bias=False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(
            in_channels=self.dim_in,
            out_channels=self.num_filters // self.expansion_ratio,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
            bias=False
        )
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, **self.bn_params)
        self.a_relu = nn.ReLU(inplace=True)

        self.b1 = nn.Conv3d(
            in_channels=self.num_filters // self.expansion_ratio,
            out_channels=self.num_filters // self.expansion_ratio,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
            bias=False
        )
        self.b1_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, **self.bn_params)
        self.b1_relu = nn.ReLU(inplace=True)

        self.b2 = nn.Conv3d(
            in_channels=self.num_filters // self.expansion_ratio,
            out_channels=self.num_filters // self.expansion_ratio,
            kernel_size=(self.kernel_size[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.kernel_size[0] // 2, 0, 0),
            bias=False
        )
        self.b2_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio, **self.bn_params)
        self.b2_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(
            in_channels=self.num_filters // self.expansion_ratio,
            out_channels=self.num_filters,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
            bias=False
        )
        self.c_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)

    def forward(self, x):
        if self.branch_style == 'simple_block':
            x = self.a1(x)
            x = self.a1_bn(x)
            x = self.a1_relu(x)

            x = self.a2(x)
            x = self.a2_bn(x)
            x = self.a2_relu(x)

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            return x
        elif self.branch_style == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b1(x)
            x = self.b1_bn(x)
            x = self.b1_relu(x)

            x = self.b2(x)
            x = self.b2_bn(x)
            x = self.b2_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x
