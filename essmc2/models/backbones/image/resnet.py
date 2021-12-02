# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from essmc2.models.registry import BACKBONES
from .resnet_impl import resnet18, resnet34, resnet50, resnet101, resnet152


@BACKBONES.register_function("ResNet")
def resnet(depth=50):
    depth_mapper = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152
    }
    cons_func = depth_mapper.get(depth)
    if cons_func is None:
        raise KeyError(f"Unsupported depth for resnet, {depth}")
    model = cons_func(pretrained=False)
    return model
