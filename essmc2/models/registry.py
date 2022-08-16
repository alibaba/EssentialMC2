# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from ..utils.file_systems import FS
from ..utils.model import load_pretrained
from ..utils.registry import Registry
from ..utils.registry import build_from_config
from ..utils.logger import get_logger


def build_model(cfg, registry, **kwargs):
    """ After build model, load pretrained model if exists key `pretrain`.

    pretrain (str, dict): Describes how to load pretrained model.
        str, treat pretrain as model path;
        dict: should contains key `path`, and other parameters token by function load_pretrained();
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be type dict, got {type(cfg)}")
    if "pretrain" in cfg:
        pretrain_cfg = cfg.pop("pretrain")
        if pretrain_cfg is not None:
            if not isinstance(pretrain_cfg, (dict, str)):
                raise TypeError(f"pretrain parameter must be a string or a dict")
    else:
        pretrain_cfg = None

    model = build_from_config(cfg, registry, **kwargs)
    # Load pretrain parameters in model
    if pretrain_cfg is None:
        return model
    if isinstance(pretrain_cfg, str):
        pretrain_cfg = {
            "path": pretrain_cfg
        }
    else:
        if "path" not in pretrain_cfg:
            raise KeyError("Expected key path in pretrain dict")

    # Get the model to local file
    path = pretrain_cfg.pop("path")
    load_pretrained(model, path, logger=get_logger(), **pretrain_cfg)

    return model


MODELS = Registry("MODELS", build_func=build_model)

BACKBONES = Registry("BACKBONES", build_func=build_model)
NECKS = Registry("NECKS", build_func=build_model)
HEADS = Registry("HEADS", build_func=build_model)
BRICKS = Registry("BRICKS", build_func=build_model)
STEMS = BRICKS
LOSSES = Registry("LOSSES", build_func=build_model)
