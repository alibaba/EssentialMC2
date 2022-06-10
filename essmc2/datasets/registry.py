# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.


from ..utils.registry import Registry, build_from_config
from ..utils.typing import check_dict_of_str_dict


def build_dataset(cfg, registry, **kwargs):
    """
    Except for ordinal dataset config, if passing a list of dataset config, then return the concat type of it
    """
    if isinstance(cfg, list):
        if len(cfg) == 0:
            raise ValueError("Dataset config contains nothing")
        if len(cfg) == 1:
            return build_from_config(cfg[0], registry, **kwargs)
        from .concat_dataset import ConcatDataset
        return ConcatDataset(*cfg)
    elif check_dict_of_str_dict(cfg, contains_type=True):
        from .concat_dataset import ConcatDataset
        return ConcatDataset(**cfg)
    else:
        return build_from_config(cfg, registry, **kwargs)


DATASETS = Registry("DATASETS", build_func=build_dataset)
