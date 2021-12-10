# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch.nn as nn

from ..registry import HEADS


@HEADS.register_class()
class ClassifierHead(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
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

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        out = self.out(x)

        return out


@HEADS.register_class()
class VideoClassifierHeadx2(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0.5):
        super(VideoClassifierHeadx2, self).__init__()
        self.dim = dim
        assert type(num_classes) is tuple
        assert len(num_classes) == 2
        self.num_classes = num_classes

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

    def forward(self, x):
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x

        out1 = self.linear1(out)
        out2 = self.linear2(out)

        return out1, out2


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

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

    def forward(self, x):
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, "pre_logits"):
            out = self.pre_logits(out)
        out = self.linear(out)

        return out


@HEADS.register_class()
class TransformerHeadx2(nn.Module):
    def __init__(self,
                 dim,
                 num_classes,
                 dropout_rate=0.5,
                 pre_logits=False):
        super(TransformerHeadx2, self).__init__()
        self.dim = dim
        assert type(num_classes) is tuple
        assert len(num_classes) == 2
        self.num_classes = num_classes

        if pre_logits:
            self.pre_logits1 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))
            self.pre_logits2 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, dim)),
                ('act', nn.Tanh())
            ]))

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.linear1 = nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = nn.Linear(dim, num_classes[1], bias=True)

    def forward(self, x):
        if hasattr(self, "dropout"):
            out = self.dropout(x)
        else:
            out = x

        if hasattr(self, "pre_logits1"):
            out1 = self.pre_logits1(out)
            out2 = self.pre_logits2(out)
        else:
            out1, out2 = out, out

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        return out1, out2
