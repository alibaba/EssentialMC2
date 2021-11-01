# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .backward import BackwardHook
from .checkpoint import CheckpointHook
from .hook import Hook
from .log import LogHook, TensorboardLogHook
from .lr import LrHook
from .sampler import DistSamplerHook
from .registry import HOOKS

"""
Normally, hooks have priorities, below we recommend priority that runs fine (low score MEANS high priority)
BackwardHook: 0
LogHook: 100
LrHook: 200
CheckpointHook: 300
SamplerHook: 400

Recommend sequences in training are:
before solve:
    TensorboardLogHook: prepare file handler
    CheckpointHook: resume checkpoint
    
before epoch:
    LogHook: clear epoch variables
    DistSamplerHook: change sampler seed

before iter:
    LogHook: record data time

after iter:
    BackwardHook: network backward
    LogHook: log
    TensorboardLogHook: log
    
after epoch:
    LrHook: reset learning rate
    CheckpointHook: save checkpoint
    
after solve:
    TensorboardLogHook: close file handler
"""

__all__ = ['HOOKS',
           'BackwardHook', 'CheckpointHook', 'Hook', 'LrHook', 'LogHook', 'TensorboardLogHook',
           'DistSamplerHook']
