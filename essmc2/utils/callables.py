# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

import collections.abc
import copy
import inspect
from functools import partial

from .registry import Registry


def build_callable(cfg, registry, **kwargs):
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
        if not issubclass(req_type_entry, collections.abc.Callable):
            raise Exception(f"{req_type_entry} type has not implemented __call__ function")
        try:
            return req_type_entry(**cfg)
        except Exception as e:
            raise Exception(f"Failed to init class {req_type_entry}, with {e}")
    elif inspect.isfunction(req_type_entry):
        try:
            return partial(req_type_entry, **cfg)
        except Exception as e:
            raise Exception(f"Failed to partial function {req_type_entry}, with {e}")
    else:
        raise TypeError(f"type must be str or class, got {type(req_type_entry)}")


CALLABLES = Registry("CALLABLES", build_func=build_callable)
