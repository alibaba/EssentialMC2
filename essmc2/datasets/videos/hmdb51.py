# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp

import numpy as np

from .base_video_dataset import BaseVideoDataset
from ..registry import DATASETS

HMDB51_LABELS = [
    "brush_hair", "cartwheel", "catch", "chew", "clap",
    "climb_stairs", "climb", "dive", "draw_sword", "dribble",
    "drink", "eat", "fall_floor", "fencing", "flic_flac",
    "golf", "handstand", "hit", "hug", "jump",
    "kick_ball", "kick", "kiss", "laugh", "pick",
    "pour", "pullup", "punch", "push", "pushup",
    "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball",
    "shoot_bow", "shoot_gun", "sit", "situp", "smile",
    "smoke", "somersault", "stand", "swing_baseball", "sword_exercise",
    "sword", "talk", "throw", "turn", "walk",
    "wave",
]


def _load_hmdb51(anno_dir, split_id=1, mode='train'):
    ret = []
    for label_name in HMDB51_LABELS:
        gt_label = HMDB51_LABELS.index(label_name)
        split_file = osp.join(anno_dir, f"{label_name}_test_split{split_id}.txt")
        with open(split_file) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                video_path = osp.join(label_name, fields[0])

                if mode == "train" and fields[1] == "1":
                    ret.append({
                        "video_path": video_path,
                        "gt_label": gt_label
                    })
                elif mode in ("eval", "test") and fields[1] == "2":
                    ret.append({
                        "video_path": video_path,
                        "gt_label": gt_label
                    })

    return ret


@DATASETS.register_class()
class Hmdb51(BaseVideoDataset):
    def __init__(self, split_id=1, **kwargs):
        kwargs['_get_samples'] = False
        super(Hmdb51, self).__init__(**kwargs)
        self.split_id = split_id

        self._get_samples()

    def _get_samples(self):
        self._samples = _load_hmdb51(self.annotation_dir, split_id=self.split_id, mode=self.mode)
