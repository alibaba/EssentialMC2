# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn
from collections import OrderedDict

from ..registry import HEADS


@HEADS.register_class()
class ClassifierHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x)


@HEADS.register_class()
class VideoClassifierHead(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0.5):
        super(VideoClassifierHead, self).__init__()
        self.dim = dim
        self.num_classes = num_classes

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.out = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x, need_features=False):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        out = self.out(x)
        if need_features:
            return out, x
        else:
            return out


@HEADS.register_class()
class TransformerHead(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0.5,
                 pre_logits=False):
        super(TransformerHead, self).__init__()
        self.dim = dim
        self.num_classes = num_classes

        if pre_logits:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.linear = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x, need_features=False):
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        out = self.linear(out)

        if need_features:
            return out, x
        else:
            return out
