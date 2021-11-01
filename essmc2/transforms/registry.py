# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from ..utils.registry import Registry, build_from_config


def build_pipeline(pipeline, registry):
    if isinstance(pipeline, list):
        if len(pipeline) == 0:
            return build_from_config(dict(type="Identity"), registry)
        elif len(pipeline) == 1:
            return build_pipeline(pipeline[0], registry)
        else:
            return build_from_config(dict(type='Compose', transforms=pipeline), registry)
    elif isinstance(pipeline, dict):
        return build_from_config(pipeline, registry)
    elif pipeline is None:
        return build_from_config(dict(type='Identity'), registry)
    else:
        raise TypeError(f"Expect pipeline_cfg to be dict or list or None, got {type(pipeline)}")


TRANSFORMS = Registry("TRANSFORMS", build_func=build_pipeline)
