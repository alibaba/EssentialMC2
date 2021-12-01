# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from ..registry import NECKS


@NECKS.register_class()
class Identity(nn.Module):
    def forward(self, inputs):
        return inputs
