# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch

from ..registry import LOSSES


@LOSSES.register_function("CrossEntropy")
def get_cross_entropy(*args, **kwargs):
    return torch.nn.CrossEntropyLoss(*args, **kwargs)
