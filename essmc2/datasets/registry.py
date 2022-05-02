# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from torch.utils.data import ConcatDataset

from ..utils.registry import Registry, build_from_config


def build_dataset(cfg, registry, **kwargs):
    """
    Except for ordinal dataset config, if passing a list of dataset config, then return the concat type of it
    """
    if isinstance(cfg, list):
        if len(cfg) == 0:
            raise ValueError("Dataset config contains nothing")
        if len(cfg) == 1:
            return build_from_config(cfg[0], registry, **kwargs)

        datasets = []
        for ele_cfg in cfg:
            dataset = build_from_config(ele_cfg, registry, **kwargs)
            datasets.append(dataset)

        return ConcatDataset(datasets)

    else:
        return build_from_config(cfg, registry, **kwargs)


DATASETS = Registry("DATASETS", build_func=build_dataset)
