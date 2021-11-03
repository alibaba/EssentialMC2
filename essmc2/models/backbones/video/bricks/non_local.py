# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

""" NonLocal block. """

import torch
import torch.nn as nn
import torch.nn.functional as F

from essmc2.models.registry import BRICKS
from .visualize_3d_module import Visualize3DModule


@BRICKS.register_class()
class NonLocal(Visualize3DModule):
    """
    Non-local block.

    See Xiaolong Wang et al.
    Non-local Neural Networks.
    """

    def __init__(self,
                 dim_in,
                 num_filters,
                 bn_params=None,
                 **kwargs):
        super(NonLocal, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.num_filters = num_filters
        self.dim_middle = self.dim_in // 2

        self.qconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0
        )

        self.kconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0
        )

        self.vconv = nn.Conv3d(
            self.dim_in,
            self.dim_middle,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0
        )

        self.out_conv = nn.Conv3d(
            self.dim_middle,
            self.num_filters,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
        )

        bn_params = bn_params or dict()
        bn_params["eps"] = 1e-5

        self.out_bn = nn.BatchNorm3d(self.num_filters, **bn_params)

    def forward(self, x):
        n, c, t, h, w = x.shape

        query = self.qconv(x).view(n, self.dim_middle, -1)
        key = self.kconv(x).view(n, self.dim_middle, -1)
        value = self.vconv(x).view(n, self.dim_middle, -1)

        attn = torch.einsum("nct,ncp->ntp", (query, key))
        attn = attn * (self.dim_middle ** -0.5)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("ntg,ncg->nct", (attn, value))
        out = out.view(n, self.dim_middle, t, h, w)
        out = self.out_conv(out)
        out = self.out_bn(out)
        return x + out
