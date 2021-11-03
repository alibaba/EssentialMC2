# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.


import os
import re
from collections import OrderedDict

import torch
from torch.utils.model_zoo import load_url as load_state_dict_from_url


def move_model_to_cpu(params):
    cpu_params = OrderedDict()
    for key, val in params.items():
        cpu_params[key] = val.cpu()
    return cpu_params


def load_pretrained(model: torch.nn.Module, path: str, map_location="cpu", logger=None):
    if logger:
        logger.info(f"Load pretrained model [{model.__class__.__name__}] from {path}")
    if os.path.exists(path):
        # From local
        state_dict = torch.load(path, map_location)
    elif path.startswith("http"):
        # From url
        state_dict = load_state_dict_from_url(path, map_location=map_location, check_hash=False)
    else:
        raise Exception(f"Cannot find {path} when load pretrained")

    return load_pretrained_dict(model, state_dict, logger)


def load_pretrained_dict(model: torch.nn.Module, state_dict: dict, logger=None):
    """ Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible.
    3. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
    """
    revise_keys = [(r'^module\.', '')]

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']

    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)
