from ..registry import LOSSES

import torch


@LOSSES.register_function("CrossEntropy")
def get_cross_entropy(*args, **kwargs):
    return torch.nn.CrossEntropyLoss(*args, **kwargs)
