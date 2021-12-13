# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp

import numpy as np

from .base_video_dataset import BaseVideoDataset
from ..registry import DATASETS


def _load_ucf101(anno_dir, split_id=1, mode='train'):
    ret = []

    cls_map = {}
    cls_file = osp.join(anno_dir, "classInd.txt")
    with open(cls_file) as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            fields = line.split()
            cls_id = int(fields[0])
            cls_name = fields[1]
            cls_map[cls_name] = cls_id

    if mode == "train":
        anno_file = "trainlist{:02d}.txt".format(split_id)
    else:
        anno_file = "testlist{:02d}.txt".format(split_id)
    anno_file = osp.join(anno_dir, anno_file)

    with open(anno_file) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            video_path = line
            cls_name = video_path.split('/')[0]
            gt_label = cls_map[cls_name]
            ret.append({
                'video_path': video_path,
                'gt_label': gt_label
            })

    return ret


@DATASETS.register_class()
class UCF101(BaseVideoDataset):
    def __init__(self, split_id=1, **kwargs):
        kwargs['_get_samples'] = False
        super(UCF101, self).__init__(**kwargs)
        self.split_id = split_id

    def _get_samples(self):
        self._samples = _load_ucf101(self.annotation_dir, split_id=self.split_id, mode=self.mode)
