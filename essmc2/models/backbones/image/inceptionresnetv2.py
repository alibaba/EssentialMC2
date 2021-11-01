# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .inceptionresnetv2_impl import inceptionresnetv2
from essmc2.models.registry import BACKBONES
from essmc2.utils.logger import get_logger
from essmc2.utils.model import load_pretrained


@BACKBONES.register_function("InceptionResNetV2")
def resnet(use_pretrain=False, load_from=""):
    model = inceptionresnetv2(pretrained=None)
    if use_pretrain:
        load_pretrained(model, load_from, logger=get_logger())
    return model
