# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.optim.lr_scheduler as lr_mod

from ..utils.registry import Registry

SUPPORT_TYPES = (
    'StepLR', 'CyclicLR', 'LambdaLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau')


def build_lr_scheduler(optimizer, cfg, **kwargs):
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"config must contain key type, got {cfg}")

    req_type = cfg.pop("type")

    assert req_type in SUPPORT_TYPES, f"req_type should in {SUPPORT_TYPES}, got {req_type}"

    cls = getattr(lr_mod, req_type)
    return cls(optimizer, **cfg)


LR_SCHEDULERS = Registry("LR_SCHEDULERS", build_func=build_lr_scheduler)
