# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from essmc2.models.registry import BACKBONES
from .inceptionresnetv2_impl import inceptionresnetv2


@BACKBONES.register_function("InceptionResNetV2")
def get_inception():
    return inceptionresnetv2(pretrained=None)
