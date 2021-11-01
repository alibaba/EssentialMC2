# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import ABCMeta


class Hook(object, metaclass=ABCMeta):
    def __init__(self, priority=0):
        self.priority = priority

    def before_solve(self, solver):
        pass

    def after_solve(self, solver):
        pass

    def before_epoch(self, solver):
        pass

    def after_epoch(self, solver):
        pass

    def before_all_iter(self, solver):
        pass

    def before_iter(self, solver):
        pass

    def after_iter(self, solver):
        pass

    def after_all_iter(self, solver):
        pass
