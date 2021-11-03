# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from essmc2.models.registry import STEMS
from ..visualize_3d_module import Visualize3DModule


@STEMS.register_class()
class Base2DStem(Visualize3DModule):
    def __init__(self,
                 dim_in=3,
                 num_filters=64,
                 kernel_size=(1, 7, 7),
                 downsampling=True,
                 downsampling_temporal=False,
                 bn_params=None,
                 **kwargs):
        super(Base2DStem, self).__init__(**kwargs)

        self.dim_in = dim_in
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        if downsampling:
            if downsampling_temporal:
                self.stride = [2, 2, 2]
            else:
                self.stride = [1, 2, 2]
        else:
            self.stride = [1, 1, 1]

        self.bn_params = bn_params or {}

        self._construct()

    def _construct(self):
        self.a = nn.Conv3d(self.dim_in,
                           self.num_filters,
                           kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                           stride=(1, self.stride[1], self.stride[2]),
                           padding=[0, self.kernel_size[1] // 2, self.kernel_size[2] // 2],
                           bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a_relu(self.a_bn(self.a(x)))
