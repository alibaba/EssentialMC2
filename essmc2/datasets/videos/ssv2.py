# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp

import numpy as np

from .base_video_dataset import BaseVideoDataset
from ..registry import DATASETS


def _load_ssv2(anno_dir, mode='train', video_format="webm"):
    cls_file = osp.join(anno_dir, "something-something-v2-labels.json")
    with open(cls_file) as f:
        cls_map = json.load(f)

    if mode == "train":
        anno_file = "something-something-v2-train.json"
    elif mode == "eval":
        anno_file = "something-something-v2-validation.json"
    else:
        anno_file = "something-something-v2-test.json"

    with open(anno_file) as f:
        videos = json.load(f)
        for video in videos:
            video_path = "{}.{}".format(video['id'], video_format)
            video['video_path'] = video_path

            if 'template' in video:
                templ = video['template'].replace('[', '').replace(']', '')
                gt_label = int(cls_map[templ])
                video['gt_label'] = gt_label

    return videos


@DATASETS.register_class()
class SSV2(BaseVideoDataset):
    def __init__(self, video_format="webm", **kwargs):
        super(SSV2, self).__init__(**kwargs)
        self.video_format = video_format

    def _get_samples(self):
        self._samples = _load_ssv2(self.annotation_dir, mode=self.mode, video_format=self.video_format)

    def _get(self, index: int):
        item = super()._get(index)
        item['meta']['id'] = self._samples[index]['id']

        return item


