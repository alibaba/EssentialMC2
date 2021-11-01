# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .registry import TRANSFORMS


@TRANSFORMS.register_class()
class Identity(object):
    def __init__(self):
        pass

    def __call__(self, item):
        return item
