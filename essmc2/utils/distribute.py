# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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
        torch.cuda.set_device(rank)


def gather_gpu_tensors(tensor):
    assert dist.get_backend() == "nccl"

    rank, world_size = get_dist_info()

    shape_tensor = torch.tensor(tensor.shape[0], device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max(dim=0)

    tensor_send = torch.zeros((shape_max, *tensor.shape[1:]), dtype=tensor.dtype, device="cuda")
    tensor_send[0:tensor.shape[0]] = tensor
    tensor_list = [torch.zeros_like(tensor_send) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_send)

    if rank == 0:
        tensors_out = []
        for tensor_recv, shape_recv in zip(tensor_list, shape_list):
            tensors_out.append(tensor_recv[0: shape_recv])
        tensor_out = torch.cat(tensors_out).contiguous()
        return tensor_out
    
