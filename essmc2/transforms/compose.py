# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torchvision.transforms as tv_transforms

from .registry import TRANSFORMS


@TRANSFORMS.register_function('Compose')
def compose(transforms):
    """ Build transforms and compose all.

    Args:
        transforms (List[dict]): List of transform configurations to be built.

    Returns:
        A callable.
    """
    trans_objs = [TRANSFORMS.build(t) for t in transforms]
    return tv_transforms.Compose(trans_objs)
