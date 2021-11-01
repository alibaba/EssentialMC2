# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch
from torch.distributed import is_initialized, get_rank, get_world_size, init_process_group


def get_dist_info():
    if is_initialized():
        return get_rank(), get_world_size()
    else:
        return 0, -1


def init_dist(backend='nccl', launcher="pytorch"):
    init_process_group(backend=backend)
    if launcher == "pytorch":
        rank, _ = get_dist_info()
        torch.cuda.set_device(rank)
