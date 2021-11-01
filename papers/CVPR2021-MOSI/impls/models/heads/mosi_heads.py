# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from essmc2.models import HEADS


@HEADS.register_class()
class MoSIHead(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0.5):
        super(MoSIHead, self).__init__()
        self.dim = dim
        self.num_classes = num_classes

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.out_joint = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x, need_features=False):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        out = self.out_joint(x)
        if need_features:
            return out, x
        else:
            return out
