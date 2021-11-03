# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp

import numpy as np

from essmc2.datasets import BaseDataset, DATASETS


@DATASETS.register_class()
class Webvision(BaseDataset):
    def __init__(self,
                 root_dir="",
                 num_classes=50,
                 **kwargs):
        super(Webvision, self).__init__(**kwargs)
        self.root_dir = root_dir
        self.num_classes = num_classes

        self.images = []
        self.labels = []

        self._load_webvision_data()

    def _load_webvision_data(self):
        file_path = "val_filelist.txt" if self.mode == "test" else "train_filelist_google.txt"
        file_path = osp.join(self.root_dir, "info", file_path)

        prefix = "val_images_256" if self.mode == "test" else ""

        with open(file_path) as f:
            for line in f.readlines():
                line = line.strip()
                img, target = line.split()
                target = int(target)
                if target < self.num_classes:
                    self.images.append(osp.join(prefix, img))
                    self.labels.append(target)

    def __len__(self) -> int:
        return len(self.images)

    def _get(self, index: int):
        return {
            "meta": {
                "prefix": self.root_dir,
                "img_path": self.images[index],
            },
            "gt_label": np.array(self.labels[index], dtype=np.int64),
            "index": np.array(index, dtype=np.int64),
        }
