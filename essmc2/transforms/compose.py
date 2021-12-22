# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .registry import TRANSFORMS


@TRANSFORMS.register_class()
class Compose(object):
    """ Compose all transform function into one.

    Args:
        transforms (List[dict]): List of transform configs.

    """
    def __init__(self, transforms):
        self.transforms = [TRANSFORMS.build(t) for t in transforms]

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item
