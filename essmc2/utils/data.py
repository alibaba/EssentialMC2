# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch


def transfer_data_to_cuda(data_map: dict) -> dict:
    """ Transfer tensors in data_map to current default gpu device
    Only tensors in dict will be transferred, list or tuple type is not supported, even in dict.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    import platform
    if platform.system() == "Darwin":
        return data_map
    ret = {}
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                ret[key] = value
            else:
                ret[key] = value.cuda(non_blocking=True)
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_cuda(value)
    return ret
