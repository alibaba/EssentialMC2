# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import warnings

from .hook import Hook
from .registry import HOOKS

_DEFAULT_BACKWARD_PRIORITY = 0


@HOOKS.register_class()
class BackwardHook(Hook):
    def __init__(self, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_BACKWARD_PRIORITY
        super(BackwardHook, self).__init__(priority=priority)

    def after_iter(self, solver):
        if solver.optimizer is not None \
                and solver.is_train_mode:
            if solver.loss is None:
                warnings.warn("solver.loss should not be None in train mode, remember to call solver._reduce_scalar()!")
                return

            if isinstance(solver.optimizer, dict):
                for _, value in solver.optimizer.items():
                    value.zero_grad()
            else:
                solver.optimizer.zero_grad()

            solver.loss.backward()

            if isinstance(solver.optimizer, dict):
                for _, value in solver.optimizer.items():
                    value.step()
            else:
                solver.optimizer.step()
            solver.loss = None
