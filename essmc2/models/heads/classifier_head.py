# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

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
