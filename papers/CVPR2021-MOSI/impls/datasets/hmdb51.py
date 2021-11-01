# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numpy as np

from essmc2.datasets import DATASETS
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_class()
class Hmdb51(BaseVideoDataset):

    def __init__(self,
                 data_root_dir,
                 annotation_dir,
                 **kwargs):
        super(Hmdb51, self).__init__(data_root_dir, annotation_dir, **kwargs)

    def _get_dataset_list_name(self):
        name = "hmdb51_{}_list.txt".format("train" if self.mode == "train" else "test")
        return name

    def _get(self, index):
        video_path, class_ = self._samples[index].strip().split(' ')
        class_ = int(class_)
        return {
            "meta": {
                "prefix": self.data_root_dir,
                "video_path": video_path
            },
            "gt_label": np.array(class_, dtype=np.int64)
        }
