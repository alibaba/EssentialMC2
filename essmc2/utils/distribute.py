# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch
import torch.distributed as dist


def get_dist_info():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, -1


def init_dist(backend='nccl', launcher="pytorch"):
    dist.init_process_group(backend=backend)
    if launcher == "pytorch":
        rank, _ = get_dist_info()
        torch.cuda.set_device(rank % torch.cuda.device_count())


def gather_data(data):
    """ Gather tensors and other picklable objects to rank 0.
    Will recursively walk through inner list and dict values.

    Args:
        data (any): Anything.

    Returns:
        A object has same structure with input `data`.
    """
    if isinstance(data, torch.Tensor):
        return gather_gpu_tensors(data)
    elif isinstance(data, dict):
        # Keep in order, dict type DO NOT guarantee a fixed key order
        keys = sorted(list(data.keys()))
        ret = OrderedDict()
        for key in keys:
            ret[key] = gather_data(data[key])
        return ret
    elif isinstance(data, list):
        return gather_list(data)
    else:
        return gather_picklable(data)


def gather_list(data):
    """ Gather list of picklable objects to a new list on rank 0.
    Will NOT recursively walk through.

    Args:
        data (list): List of picklable things.

    Returns:
        A new flat list.
    """
    rank, _ = get_dist_info()
    list_of_list = gather_picklable(data)
    if rank == 0:
        return sum(list_of_list, [])


def gather_picklable(data):
    """ Gather picklable object to a list on rank 0.
    Will NOT recursively walk through.

    Args:
        data (picklable): Picklable data.

    Returns:
        A list contains data collected.
    """
    from packaging import version
    from torch.version import __version__
    if version.parse(__version__) < version.parse("1.8.0"):
        return _gather_picklable_custom(data)
    else:
        rank, world_size = get_dist_info()
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, data)
        if rank == 0:
            return obj_list


def _gather_picklable_custom(data):
    """ Custom implementation function to gather picklable object to a list on rank 0.
    If torch version is lower than 1.8.0, use this.

    Args:
        data (picklable): Picklable data.

    Returns:
        A list contains data collected.
    """
    import pickle
    byte_tensor = torch.tensor(bytearray(pickle.dumps(data)), dtype=torch.uint8, device='cuda')
    rank, world_size = get_dist_info()
    shape_tensor = torch.tensor(byte_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    tensor_send = torch.zeros(shape_max, dtype=byte_tensor.dtype, device="cuda")
    tensor_send[0:shape_tensor[0]] = byte_tensor
    tensor_list = [torch.zeros_like(tensor_send) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_send)

    if rank == 0:
        data_out = []
        for tensor_recv, shape_recv in zip(tensor_list, shape_list):
            new_data = pickle.loads(tensor_recv[:shape_recv[0]].cpu().numpy().tobytes())
            data_out.append(new_data)
        return data_out


def gather_gpu_tensors(tensor):
    """ Gather tensor to rank 0 and concat it.

    Args:
        tensor (torch.Tensor):

    Returns:
        A new tensor.
    """
    assert dist.get_backend() == "nccl"

    device = tensor.device
    if device.type == 'cpu':
        tensor = tensor.cuda()

    rank, world_size = get_dist_info()

    shape_tensor = torch.tensor(tensor.shape[0], device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    tensor_send = torch.zeros((shape_max, *tensor.shape[1:]), dtype=tensor.dtype, device="cuda")
    tensor_send[0:tensor.shape[0]] = tensor
    tensor_list = [torch.zeros_like(tensor_send) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_send)

    if rank == 0:
        tensors_out = []
        for tensor_recv, shape_recv in zip(tensor_list, shape_list):
            tensors_out.append(tensor_recv[0: shape_recv])
        tensor_out = torch.cat(tensors_out).contiguous()
        if device.type == 'cpu':
            tensor_out = tensor_out.cpu()
        return tensor_out
