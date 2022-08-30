# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
import numbers
import os
import os.path as osp
from typing import Optional

import numpy as np
import torch
import torch.distributed as du

from .hook import Hook
from .registry import HOOKS
from ..utils.file_systems import FS

_DEFAULT_TENSORBOARD_PRIORITY = 101


def _print_dict(kvs: dict) -> list:
    ret = []
    for key, value in kvs.items():
        if isinstance(value, dict):
            res = _print_dict(value)
            for prefix, vv in res:
                prefix.insert(0, key)
                ret.append((prefix, vv))
        else:
            ret.append(([key], value))
    return ret


@HOOKS.register_class()
class TensorboardLogHook(Hook):
    def __init__(self, log_dir=None, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_TENSORBOARD_PRIORITY
        super(TensorboardLogHook, self).__init__(priority=priority)
        self.log_dir = log_dir
        self._local_log_dir = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            import warnings
            warnings.warn(f"You may run `pip install tensorboard` to use {self.__class__.__name__}")
            exit(-1)
        self.writer: Optional[SummaryWriter] = None

    def before_solve(self, solver):
        from torch.utils.tensorboard import SummaryWriter
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, "tensorboard")

        self._local_log_dir, _ = FS.map_to_local(self.log_dir)
        os.makedirs(self._local_log_dir, exist_ok=True)
        self.writer = SummaryWriter(self._local_log_dir)
        solver.logger.info(f"Tensorboard: save to {self.log_dir}")

    def after_iter(self, solver):
        if self.writer is None:
            return
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        for key, value in outputs.items():
            if key == "batch_size":
                continue
            if isinstance(value, torch.Tensor):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = value.item()
            elif isinstance(value, np.ndarray):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = float(value)
            elif isinstance(value, numbers.Number):
                # Must be number
                pass
            else:
                continue

            self.writer.add_scalar(f"{mode}/iter/{key}", value, global_step=solver.total_iter)

    def after_epoch(self, solver):
        if self.writer is None:
            return
        outputs = solver.epoch_outputs.copy()
        ret = _print_dict(outputs)
        for prefix, value in ret:
            prefix.insert(1, 'epoch')
            prefix_str = '/'.join(prefix)
            self.writer.add_scalar(prefix_str, value, global_step=solver.epoch)

        self.writer.flush()
        # Put to remote file systems every epoch
        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_solve(self, solver):
        if self.writer is None:
            return
        if self.writer:
            self.writer.close()

        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)
