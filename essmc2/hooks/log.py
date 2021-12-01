# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numbers
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as du
from torch.utils.tensorboard import SummaryWriter

from .hook import Hook
from .registry import HOOKS
from ..utils.file_systems import FS, LocalFs
from ..utils.logger import LogAgg

_DEFAULT_LOG_PRIORITY = 100


def _format_float(x):
    if abs(x) - int(abs(x)) < 0.01:
        return "{:.6f}".format(x)
    else:
        return "{:.4f}".format(x)


def _print_v(x):
    if isinstance(x, float):
        return _format_float(x)
    elif isinstance(x, torch.Tensor) and x.ndim == 0:
        return _print_v(x.item())
    else:
        return f"{x}"


def _print_iter_log(solver, outputs, final=False):
    extra_vars = solver.collect_log_vars()
    outputs.update(extra_vars)
    s = []
    for k, v in outputs.items():
        if k in ('data_time', 'time'):
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            s.append(f"{k}: " + _print_v(v[0]) + f"({_print_v(v[1])})")
        else:
            s.append(f"{k}: " + _print_v(v))
    if "time" in outputs:
        v = outputs["time"]
        s.insert(0,
                 f"time: "
                 + _print_v(v[0])
                 + f"({_print_v(v[1])})"
                 )
    if "data_time" in outputs:
        v = outputs["data_time"]
        s.insert(0,
                 f"data_time: "
                 + _print_v(v[0])
                 + f"({_print_v(v[1])})"
                 )
    solver.logger.info(
        f"Epoch [{solver.epoch}/{solver.max_epochs}], "
        f"iter: [{solver.iter + 1 if not final else solver.iter}/{solver.epoch_max_iter}], "
        f"{', '.join(s)}")


@HOOKS.register_class()
class LogHook(Hook):
    def __init__(self, log_interval=10, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_LOG_PRIORITY
        super(LogHook, self).__init__(priority=priority)
        self.log_interval = log_interval
        self.log_agg_dict = defaultdict(LogAgg)

        self.last_log_step = ("train", 0)

        self.time = time.time()
        self.data_time = 0

    def before_all_iter(self, solver):
        self.time = time.time()
        self.last_log_step = (solver.mode, 0)

    def before_iter(self, solver):
        data_time = time.time() - self.time
        self.data_time = data_time

    def after_iter(self, solver):
        log_agg = self.log_agg_dict[solver.mode]
        iter_time = time.time() - self.time
        self.time = time.time()
        outputs = solver.iter_outputs.copy()
        outputs["time"] = iter_time
        outputs["data_time"] = self.data_time
        if "batch_size" in outputs:
            batch_size = outputs.pop("batch_size")
        else:
            batch_size = 1
        log_agg.update(outputs, batch_size)
        if (solver.iter + 1) % self.log_interval == 0:
            _print_iter_log(solver, log_agg.aggregate(self.log_interval))
            self.last_log_step = (solver.mode, solver.iter + 1)

    def after_all_iter(self, solver):
        outputs = self.log_agg_dict[solver.mode].aggregate(solver.iter - self.last_log_step[1])
        solver.agg_iter_outputs = {key: value[1] for key, value in outputs.items()}
        current_log_step = (solver.mode, solver.iter)
        if current_log_step != self.last_log_step:
            _print_iter_log(solver, outputs, final=True)
            self.last_log_step = current_log_step

        for _, value in self.log_agg_dict.items():
            value.reset()

    def after_epoch(self, solver):
        outputs = solver.epoch_outputs
        mode_s = []
        for mode_name, kvs in outputs.items():
            if len(kvs) == 0:
                return
            s = [f"{k}: " + _print_v(v) for k, v in kvs.items()]
            mode_s.append(f"{mode_name} -> {', '.join(s)}")
        if len(mode_s) > 1:
            states = '\n\t'.join(mode_s)
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], \n\t'
                f'{states}'
            )
        elif len(mode_s) == 1:
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], {mode_s[0]}')


@HOOKS.register_class()
class TensorboardLogHook(Hook):
    def __init__(self, log_dir=None, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_LOG_PRIORITY
        super(TensorboardLogHook, self).__init__(priority=priority)
        self.log_dir = log_dir
        self._local_log_dir = None
        self.writer: Optional[SummaryWriter] = None

    def before_solve(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, "tensorboard")
        tb_client = FS.get_fs_client(self.log_dir)

        if type(tb_client) is LocalFs:
            self.writer = SummaryWriter(self.log_dir)
        else:
            local_tb_dir = tb_client.convert_to_local_path(self.log_dir)
            os.makedirs(local_tb_dir, exist_ok=True)
            self._local_log_dir = local_tb_dir
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
        for mode, kvs in outputs.items():
            for key, value in kvs.items():
                self.writer.add_scalar(f"{mode}/epoch/{key}", value, global_step=solver.epoch)

        self.writer.flush()
        # Put to remote file systems every epoch
        tb_client = FS.get_fs_client(self.log_dir)
        if type(tb_client) is not LocalFs and self._local_log_dir is not None:
            tb_client.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_solve(self, solver):
        if self.writer is None:
            return
        if self.writer:
            self.writer.close()

        tb_client = FS.get_fs_client(self.log_dir)
        if type(tb_client) is not LocalFs and self._local_log_dir is not None:
            tb_client.put_dir_from_local_dir(self._local_log_dir, self.log_dir)