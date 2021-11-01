# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .resnet_impl import resnet18, resnet34, resnet50, resnet101, resnet152
from essmc2.models.registry import BACKBONES
from essmc2.utils.logger import get_logger
from essmc2.utils.model import load_pretrained


@BACKBONES.register_function("ResNet")
def resnet(depth=50, use_pretrain=False, load_from=""):
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
    if use_pretrain:
        load_pretrained(model, load_from, logger=get_logger())
    return model
