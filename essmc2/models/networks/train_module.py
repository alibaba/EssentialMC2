# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class TrainModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(TrainModule, self).__init__()

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, *inputs, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, *inputs, **kwargs):
        pass
