# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import numpy as np
import torch

from .file_systems import FS


def worker_init_fn(worker_id, seed=None, worker_device=None, file_systems=None):
    """ Init dataloader worker.
    0. set random seed for numpy module which will not be set by pytorch automatically;
    1. set data worker cuda device;
    2. set file systems in worker context;

    Args:
        worker_id (int): Dataloader worker id.
        seed (Optional[int]): Indicates if a random is given by user.
        worker_device (Optional[str]): Dataloader worker default cuda device.
        file_systems (Optional[Union[dict, List[dict]]]: File system config.

    Returns:

    """
    if seed is not None:
        # torch.manual_seed/random.seed
        # torch.cuda.manual_seed/torch.cuda.manual_seed_all
        # and all other env variables
        # are automatically set in either fork or spawn multiprocessing mode
        np.random.seed(torch.initial_seed() % 2 ** 32)
    if worker_device is not None:
        torch.cuda.set_device(worker_device)
    FS.init_fs_client(file_systems)


def transfer_data_to_numpy(data_map: dict) -> dict:
    """ Transfer tensors in data_map to numpy type.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_numpy(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_numpy(t) for t in value])
        else:
            ret[key] = value
    return ret


def transfer_data_to_cpu(data_map: dict) -> dict:
    """ Transfer tensors in data_map to cpu device.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.detach().cpu()
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_cpu(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_cpu(t) for t in value])
        else:
            ret[key] = value
    return ret


def transfer_data_to_cuda(data_map: dict) -> dict:
    """ Transfer tensors in data_map to current default gpu device.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    import platform
    if platform.system() == "Darwin":
        return data_map
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                ret[key] = value
            else:
                ret[key] = value.cuda(non_blocking=True)
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_cuda(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_cuda(t) for t in value])
        else:
            ret[key] = value
    return ret
