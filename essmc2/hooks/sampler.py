# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .hook import Hook
from .registry import HOOKS

_DEFAULT_SAMPLER_PRIORITY = 400


@HOOKS.register_class()
class DistSamplerHook(Hook):
    """ DistributedDataSampler needs to set_epoch to shuffle sample indexes
    """

    def __init__(self, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_SAMPLER_PRIORITY
        super(DistSamplerHook, self).__init__(priority=priority)

    def before_epoch(self, solver):
        for name, data_loader in solver.data_loaders.items():
            if name == "train":
                if hasattr(data_loader.sampler, "set_epoch"):
                    solver.logger.info(f"distribute sampler set_epoch to {solver.epoch}")
                    data_loader.sampler.set_epoch(solver.epoch)
                elif hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                    solver.logger.info(f"distribute sampler set_epoch to {solver.epoch}")
                    data_loader.batch_sampler.sampler.set_epoch(solver.epoch)
