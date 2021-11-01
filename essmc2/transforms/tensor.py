# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numpy as np
import torch

from .registry import TRANSFORMS


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"Unsupported type {type(data)}")


@TRANSFORMS.register_class()
class ToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, item):
        for key in self.keys:
            item[key] = to_tensor(item[key])
        return item


@TRANSFORMS.register_class()
class Select(object):
    def __init__(self, keys):
        self.keys = keys
        self.store_keys = ["meta"]

    def __call__(self, item):
        data = {}
        for key in self.keys:
            data[key] = item[key]
        for key in self.store_keys:
            data[key] = item[key]
        return data


@TRANSFORMS.register_class()
class TensorToGPU(object):
    def __init__(self, keys, device_id=None):
        self.keys = keys
        self.device_id = device_id

    def __call__(self, item):
        ret = {}
        for key, value in item.items():
            if key in self.keys and isinstance(value, torch.Tensor) and torch.cuda.is_available():
                ret[key] = value.cuda(self.device_id, non_blocking=True)
            else:
                ret[key] = value
        return ret
