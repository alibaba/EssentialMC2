# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import logging
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as du

from essmc2.utils.file_systems import FS
from .distribute import get_dist_info


def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger


def init_logger(in_logger, log_file=None, dist_launcher="pytorch"):
    """ Add file handler to logger on rank 0 and set log level by dist_launcher

    Args:
        in_logger (logging.Logger):
        log_file (str, None): if not None, a file handler will be add to in_logger
        dist_launcher (str, None):
    """
    rank, _ = get_dist_info()
    if rank == 0:
        if log_file is not None:
            file_handler = FS.get_fs_client(log_file).get_logging_handler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            in_logger.addHandler(file_handler)
            in_logger.info(f"Running task with log file: {log_file}")
        in_logger.setLevel(logging.INFO)
    else:
        if dist_launcher == "pytorch":
            in_logger.setLevel(logging.ERROR)
        else:
            # Distribute Training with more than one machine, we'd like to show logs on every machine.
            in_logger.setLevel(logging.INFO)


class LogAgg(object):
    """ Log variable aggregate tool. Recommend to invoke clear() function after one epoch.
    In distributed training environment, tensor variable will be all reduced to get an average.

    Example:
        >>> agg = LogAgg()
        >>> agg.update(dict(loss=0.1, accuracy=0.5))
        >>> agg.update(dict(loss=0.2, accuracy=0.6))
        >>> agg.aggregate()
        >>> agg.output
        OrderedDict([('loss', 0.15000000000000002), ('accuracy', 0.55)])
    """

    def __init__(self):
        self.buffer = OrderedDict()
        self.counter = []
        self.output = OrderedDict()

    def update(self, kv: dict, count=1):
        """ Update variables

        Args:
            kv (dict): a dict with value type in (torch.Tensor, numbers)
            count (int): divider, default is 1
        """
        for k, v in kv.items():
            if isinstance(v, torch.Tensor):
                if not v.ndim == 0:
                    continue
                # Must be scalar
                if du.is_available() and du.is_initialized():
                    v = v.data.clone()
                    du.all_reduce(v.div_(du.get_world_size()))
                v = v.item()
            if k not in self.buffer:
                self.buffer[k] = []
            self.buffer[k].append(v)
        self.counter.append(count)

    def aggregate(self, n=0):
        """ Do aggregation.

        Args:
            n (int): recent n numbers, if 0, start from 0
        """
        for key in self.buffer:
            values = np.array(self.buffer[key][-n:])
            nums = np.array(self.counter[-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg

    def clear(self):
        self.buffer.clear()
        self.counter.clear()
        self.output.clear()
