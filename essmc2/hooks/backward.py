# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .hook import Hook
from .registry import HOOKS

_DEFAULT_BACKWARD_PRIORITY = 0


@HOOKS.register_class()
class BackwardHook(Hook):
    def __init__(self, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_BACKWARD_PRIORITY
        super(BackwardHook, self).__init__(priority=priority)

    def after_iter(self, solver):
        if solver.optimizer is not None and solver.is_train_mode and 'loss' in solver.iter_outputs:
            solver.optimizer.zero_grad()
            solver.iter_outputs["loss"].backward()
            solver.optimizer.step()
