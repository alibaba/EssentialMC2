# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from essmc2.models.registry import BACKBONES
from essmc2.utils.logger import get_logger
from essmc2.utils.model import load_pretrained
from .preresnet_impl import preresnet_18, preresnet_34, preresnet_50, preresnet_101, preresnet_152


@BACKBONES.register_function("PreResNet")
def preresnet(depth=18, use_pretrain=False, load_from=""):
    depth_mapper = {
        18: preresnet_18,
        34: preresnet_34,
        50: preresnet_50,
        101: preresnet_101,
        152: preresnet_152
    }
    cons_func = depth_mapper.get(depth)
    if cons_func is None:
        raise KeyError(f"Unsupported depth for preresnet, support {depth_mapper.keys()}, got {depth}")
    model = cons_func()
    if use_pretrain:
        load_pretrained(model, load_from, logger=get_logger())
    return model
