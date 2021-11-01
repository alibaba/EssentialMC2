# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from essmc2.models.registry import STEMS
from .base_3d_stem import Base3DStem


@STEMS.register_class()
class DownSampleStem(Base3DStem):
    def __init__(self,
                 **kwargs):
        super(DownSampleStem, self).__init__(**kwargs)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.maxpool(self.a_relu(self.a_bn(self.a(x))))
