# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import math
import time
from collections import defaultdict

import torch

from .hook import Hook
from .registry import HOOKS
from ..utils.logger import LogAgg

_DEFAULT_LOG_PRIORITY = 100


def _format_float(x):
    if math.isinf(x):
        return 'inf'
    if math.isnan(x):
        return 'nan'
    if abs(x) < 1.0:
        diff = abs(x) - int(abs(x))
        if diff < 0.01:
            return "{:.6f}".format(x)
        elif diff < 0.0001:
            return "{:.8f}".format(x)

    return "{:.4f}".format(x)


def _print_v(x):
    if isinstance(x, float):
        return _format_float(x)
    elif isinstance(x, torch.Tensor) and x.ndim == 0:
        return _print_v(x.item())
    else:
        return f"{x}"


def _print_dict(kvs: dict) -> list:
    # if value is dict，show it later
    lines = []
    dict_keys = []
    normal_keys = []
    for key in kvs.keys():
        if isinstance(kvs[key], dict):
            dict_keys.append(key)
        else:
            normal_keys.append(key)

    normal_line_fields = [f"{k}: " + _print_v(kvs[k]) for k in normal_keys]

    dict_lines = []
    for key in dict_keys:
        inner_lines = _print_dict(kvs[key])
        if '->' not in inner_lines[0]:
            dict_lines.append(f"{key} -> {inner_lines[0]}")
            for s in inner_lines[1:]:
                dict_lines.append(f"  {s}")
        else:
            dict_lines.append(f"{key} -> ")
            for s in inner_lines:
                dict_lines.append(f"  {s}")

    if len(normal_line_fields) > 0:
        lines.append(", ".join(normal_line_fields))

    lines.extend(dict_lines)

    return lines


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

        output_lines = _print_dict(outputs)

        if len(output_lines) > 1:
            states = '\n  '.join(output_lines)
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], {states}'
            )
        elif len(output_lines) == 1:
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], {output_lines[0]}')
