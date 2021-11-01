# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .registry import DATASETS
from .base_dataset import BaseDataset
from .image_classify_json_dataset import ImageClassifyJsonDataset

__all__ = [
    'DATASETS',
    'BaseDataset',
    'ImageClassifyJsonDataset'
]
