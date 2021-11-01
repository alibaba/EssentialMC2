# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.optim as optim

from ..utils.registry import Registry

SUPPORT_TYPES = (
    'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'AdamW', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam')


def build_optimizer(params, cfg, **kwargs):
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"config must contain key type, got {cfg}")

    req_type = cfg.pop("type")

    assert req_type in SUPPORT_TYPES, f"req_type should in {SUPPORT_TYPES}, got {req_type}"

    cls = getattr(optim, req_type)
    return cls(params, **cfg)


OPTIMIZERS = Registry("OPTIMIZERS", build_func=build_optimizer)
