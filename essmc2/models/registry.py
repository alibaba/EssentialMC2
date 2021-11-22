# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from ..utils.registry import Registry

MODELS = Registry("MODELS")

BACKBONES = Registry("BACKBONES")
NECKS = Registry("NECKS")
HEADS = Registry("HEADS")
BRICKS = Registry("BRICKS")
STEMS = BRICKS
LOSSES = Registry("LOSSES")
