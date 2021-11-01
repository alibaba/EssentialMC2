# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import abstractmethod, ABCMeta

from .visualize_3d_module import Visualize3DModule


class BaseBranch(Visualize3DModule, metaclass=ABCMeta):
    def __init__(self,
                 branch_style="simple_block",
                 construct_branch=True,
                 **kwargs):
        super(BaseBranch, self).__init__(**kwargs)
        self.branch_style = branch_style

        if construct_branch:
            self._construct_branch()

    def _construct_branch(self):
        if self.branch_style == "simple_block":
            self._construct_simple_block()
        elif self.branch_style == "bottleneck":
            self._construct_bottleneck()

    @abstractmethod
    def _construct_simple_block(self):
        return

    @abstractmethod
    def _construct_bottleneck(self):
        return

    @abstractmethod
    def forward(self, x):
        return
