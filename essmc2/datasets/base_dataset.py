# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from essmc2.utils.file_systems import FS
from ..transforms import build_pipeline, TRANSFORMS


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 mode='test',
                 pipeline=None,
                 fs_cfg=None):
        super(BaseDataset, self).__init__()
        self.mode = mode
        self.pipeline = build_pipeline(pipeline, TRANSFORMS)
        self.fs_cfg = fs_cfg

    def __getitem__(self, index: int):
        item = self._get(index)
        return self.pipeline(item)

    @abstractmethod
    def _get(self, index: int):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: mode={self.mode}, len={len(self)}"

    def _mp_init_fs(self):
        """ In multiprocess context, file system should be inited in each worker.
        It should be invoked before invoking io op in transform pipeline.
        """
        FS.init_fs_client(self.fs_cfg)
