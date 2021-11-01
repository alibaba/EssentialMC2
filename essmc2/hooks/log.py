# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp
import time
from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as du
from torch.utils.tensorboard import SummaryWriter

from .hook import Hook
from .registry import HOOKS
from ..utils.logger import LogAgg

_DEFAULT_LOG_PRIORITY = 100


def _format_float(x):
    if abs(x) < 0.01:
        return "{:.6f}".format(x)
    else:
        return "{:.4f}".format(x)


def _print_iter_log(solver, outputs, final=False):
    extra_vars = solver.collect_log_vars()
    outputs.update(extra_vars)
    s = [f"{k}: " + (_format_float(v) if isinstance(v, float) else f"{v}") for k, v in outputs.items()
         if k not in ('data_time', 'time')]
    if "time" in outputs:
        s.insert(0, f"time: " + "{:.4f}".format(outputs["time"]))
    if "data_time" in outputs:
        s.insert(0, f"data_time: " + "{:.4f}".format(outputs["data_time"]))
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

        self.last_log_step = "train-0"

        self.time = time.time()
        self.data_time = 0

    def before_all_iter(self, solver):
        self.time = time.time()

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
            log_agg.aggregate()
            _print_iter_log(solver, log_agg.output.copy())
            self.last_log_step = f"{solver.mode}-{solver.iter + 1}"

    def after_all_iter(self, solver):
        current_log_step = f"{solver.mode}-{solver.iter}"
        if current_log_step == self.last_log_step:
            return
        log_agg = self.log_agg_dict[solver.mode]
        log_agg.aggregate()
        _print_iter_log(solver, log_agg.output.copy(), final=True)
        self.last_log_step = current_log_step

        for _, value in self.log_agg_dict.items():
            value.clear()

    def after_epoch(self, solver):
        outputs = solver.epoch_outputs
        if len(outputs) == 0:
            return
        s = [f"{k}: " + _format_float(v) if isinstance(v, float) else f"{v}" for k, v in outputs.items()]
        solver.logger.info(
            f"Epoch [{solver.epoch}/{solver.max_epochs}], "
            f"{', '.join(s)}")


@HOOKS.register_class()
class TensorboardLogHook(Hook):
    def __init__(self, log_dir=None, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_LOG_PRIORITY
        super(TensorboardLogHook, self).__init__(priority=priority)
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None

    def before_solve(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return

        if self.log_dir is None:
            i = 0
            while True:
                log_dir = osp.join(solver.work_dir, "tf_logs" if i == 0 else f"tf_logs-{i}")
                if osp.exists(log_dir):
                    i += 1
                    continue
                else:
                    break
            self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)
        solver.logger.info(f"Tensorboard: save to {self.log_dir}")

    def after_iter(self, solver):
        if self.writer is None:
            return
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                value = value.item()
            self.writer.add_scalar(f"{mode}/{key}", value, global_step=solver.total_iter)

    def after_epoch(self, solver):
        if self.writer is None:
            return
        outputs = solver.epoch_outputs.copy()
        for key, value in outputs.items():
            self.writer.add_scalar(key, value, global_step=solver.epoch)

    def after_solve(self, solver):
        if self.writer is None:
            return
        if self.writer:
            self.writer.close()
