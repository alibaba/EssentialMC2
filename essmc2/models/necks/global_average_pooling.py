# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from ..registry import NECKS


@NECKS.register_class()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.
    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
                                 f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def infer(self, x):
        if x.ndim == 2:
            return x
        return self.gap(x).view(x.size(0), -1)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            return tuple([self.infer(x) for x in inputs])
        else:
            return self.infer(inputs)
