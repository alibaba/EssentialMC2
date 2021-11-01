# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import json

import numpy as np

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_class()
class ImageClassifyJsonDataset(BaseDataset):
    """
    Dataset for image classification wrapper

    Args:
        json_path (str): json file which contains all instances, should be a list of dict
            which contains img_path and gt_label
        image_dir (str or None): image directory, if None, img_path in json_path will be considered as absolute path
        classes (list[str] or None): image class description
    """

    def __init__(self,
                 json_path,
                 image_dir=None,
                 classes=None,
                 **kwargs):
        super(ImageClassifyJsonDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.json_path = json_path
        self.classes = classes

        self.content_list = []
        self._load_json()

    def _get(self, index: int):
        item = self.content_list[index]
        ret = {
            "meta": {
                "img_path": item["img_path"]
            },
            "gt_label": np.asarray(item["gt_label"], dtype=np.int64)
        }
        if self.image_dir is not None:
            ret["meta"]["prefix"] = self.image_dir
        if self.classes is not None:
            ret["meta"]["label_name"] = self.classes[item["gt_label"]]
        return ret

    def _load_json(self):
        with open(self.json_path) as f:
            self.content_list = json.load(f)

    def __len__(self) -> int:
        return len(self.content_list)
