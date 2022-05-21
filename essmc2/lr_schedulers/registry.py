# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import warnings

import torch.optim.lr_scheduler as lr_mod

from ..utils.registry import Registry
from ..utils.typing import check_dict_of_str_dict

SUPPORT_TYPES = (
    'StepLR', 'CyclicLR', 'LambdaLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau')


def build_lr_scheduler(optimizer, cfg, **kwargs):
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")

    def _build_lr_scheduler(sub_optim, sub_cfg):
        req_type = sub_cfg.pop("type")
        assert req_type in SUPPORT_TYPES, f"req_type should in {SUPPORT_TYPES}, got {req_type}"
        cls = getattr(lr_mod, req_type)
        return cls(sub_optim, **sub_cfg)

    if isinstance(optimizer, dict):
        # if optimizer is a composite value which contains multiple optimizer
        ret = {}
        if check_dict_of_str_dict(cfg, contains_type=True):
            for module_name, lr_cfg in cfg.items():
                if module_name not in optimizer:
                    warnings.warn(f"{module_name} not in optimizer, please check optimizer and lr_scheduler configs")
                    continue
                ret[module_name] = _build_lr_scheduler(optimizer[module_name], lr_cfg)
        elif 'type' in cfg:
            for module_name, sub_optimizer in optimizer.items():
                ret[module_name] = _build_lr_scheduler(sub_optimizer, cfg.copy())
        else:
            raise KeyError(f"config must contain key type, got {cfg}")
        return ret
    else:
        if 'type' not in cfg:
            raise KeyError(f"config must contain key type, got {cfg}")
        return _build_lr_scheduler(optimizer, cfg)


LR_SCHEDULERS = Registry("LR_SCHEDULERS", build_func=build_lr_scheduler)
