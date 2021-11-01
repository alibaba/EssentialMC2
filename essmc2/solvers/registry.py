# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import copy
import inspect

from ..utils.registry import Registry


def build_solver(model, cfg, registry, **kwargs):
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"config must contain key type, got {cfg}")
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be type Registry, got {type(registry)}")

    cfg = copy.deepcopy(cfg)
    req_type = cfg.pop("type")
    req_type_entry = req_type
    if isinstance(req_type, str):
        req_type_entry = registry.get(req_type)
        if req_type_entry is None:
            raise KeyError(f"{req_type} not found in {registry.name} registry")

    if kwargs is not None:
        cfg.update(kwargs)

    if inspect.isclass(req_type_entry):
        try:
            return req_type_entry(model, **cfg)
        except Exception as e:
            raise Exception(f"Failed to init class {req_type_entry}, with {e}")
    else:
        raise TypeError(f"type must be str or class, got {type(req_type_entry)}")


SOLVERS = Registry("SOLVERS", build_func=build_solver, allow_types=("class",))
