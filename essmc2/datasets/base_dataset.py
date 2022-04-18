# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from ..transforms.registry import build_pipeline, TRANSFORMS


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 mode='test',
                 pipeline=None):
        super(BaseDataset, self).__init__()
        self.mode = mode
        self.pipeline = build_pipeline(pipeline, TRANSFORMS)

    def __getitem__(self, index: int):
        item = self._get(index)
        return self.pipeline(item)

    @abstractmethod
    def _get(self, index: int):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: mode={self.mode}, len={len(self)}"
