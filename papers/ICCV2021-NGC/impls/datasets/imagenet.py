# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os
import os.path as osp

import numpy as np

from essmc2.datasets import BaseDataset, DATASETS


@DATASETS.register_class()
class ImageNet(BaseDataset):
    def __init__(self,
                 root_dir="",
                 num_classes=50,
                 **kwargs):
        super(ImageNet, self).__init__(**kwargs)
        self.root_dir = root_dir
        self.num_classes = num_classes

        self.images = []
        self.labels = []

        self._load_imagenet_data()

    def _load_imagenet_data(self):
        dir_path = osp.join(self.root_dir, self.mode)

        for c in range(self.num_classes):
            cls_dir = osp.join(dir_path, str(c))
            img_list = [osp.join(cls_dir, t) for t in os.listdir(cls_dir) if t.endswith("JPEG")]
            for img in img_list:
                self.images.append(img)
                self.labels.append(c)

    def __len__(self) -> int:
        return len(self.images)

    def _get(self, index: int):
        return {
            "meta": {
                "img_path": self.images[index],
            },
            "gt_label": np.array(self.labels[index], dtype=np.int64),
            "index": np.array(index, dtype=np.int64),
        }
