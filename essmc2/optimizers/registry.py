# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import torch.nn
import torch.optim as optim
import torch.optim.adam

from ..utils.registry import Registry
from ..utils.typing import check_dict_of_str_dict

SUPPORT_TYPES = (
    'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'AdamW', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam')


def build_optimizer(model_or_params, cfg, **kwargs):
    """
    Args:
        model_or_params (Union[torch.nn.Module, Union[dict, list]): model or optimizer param_groups
        cfg (dict):
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")

    def _build_optimizer(sub_module_or_params, sub_cfg):
        req_type = sub_cfg.pop('type')
        assert req_type in SUPPORT_TYPES, f"req_type should in {SUPPORT_TYPES}, got {req_type}"
        cls = getattr(optim, req_type)
        if isinstance(sub_module_or_params, torch.nn.Module):
            return cls([t for t in sub_module_or_params.parameters() if t.requires_grad], **sub_cfg)
        else:
            return cls(sub_module_or_params, **sub_cfg)

    if isinstance(model_or_params, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        model_or_params = model_or_params.module

    if check_dict_of_str_dict(cfg, contains_type=True):
        is_model = isinstance(model_or_params, torch.nn.Module)
        ret = {}
        for module_name, optim_cfg in cfg.items():
            if is_model:
                if not hasattr(model_or_params, module_name):
                    raise ValueError(f"Cannot find sub module {module_name} in model.")
                module: torch.nn.Module = getattr(model_or_params, module_name)
                if not isinstance(module, torch.nn.Module):
                    raise TypeError(f"module {module_name} must be torch.nn.Module, got {type(module)}")
                ret[module_name] = _build_optimizer(module, optim_cfg)
            else:
                if module_name not in model_or_params:
                    raise ValueError(f"Cannot find sub module {module_name} in model parameters.")
                ret[module_name] = _build_optimizer(model_or_params[module_name], optim_cfg)
        return ret
    elif 'type' in cfg:
        return _build_optimizer(model_or_params, cfg)
    else:
        raise KeyError(f"config must contain key type or values should contains type, got {cfg}")


OPTIMIZERS = Registry("OPTIMIZERS", build_func=build_optimizer)
