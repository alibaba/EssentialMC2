# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import abstractmethod, ABCMeta

import numpy as np

from essmc2.datasets import BaseDataset


class BaseVideoDataset(BaseDataset, metaclass=ABCMeta):
    """ BaseVideoDataset Interface.

    In train mode,
        DecodeVideoToTensor will sample a clip from video randomly with `clip_id` and `num_clips` in T dimension.
        RandomResizedCropVideo will randomly crop a new clip from clip in H*W dimension with `spatial_id`.

    In test model, with given `clip_id`, `num_clips` and `spatial_id`,
        DecodeVideoToTensor will sample a fixed clip from video in T dimension.
        AutoResizedCropVideo will crop a fixed position new clip in H*W dimension.

    Args:
        data_root_dir (str): Location to load videos.
        annotation_dir (str): Location to load annotations.
        temporal_crops (int): A video wound be split by time into temporal clips for ensemble, default is 1.
        spatial_crops (int): A temporal clip wound be cropped in space spatial crops times for ensemble, default is 1.
            So a video wound be transformed into `num_clips` clips to ensemble, where num_clips = temporal_crops * spatial_crops.
        fix_len (int, None): For debug, sometimes we want to quickly train a model and test.

    """

    def __init__(self,
                 data_root_dir,
                 annotation_dir,
                 temporal_crops=1,
                 spatial_crops=1,
                 fix_len=None,
                 _get_samples=True,
                 **kwargs):
        super(BaseVideoDataset, self).__init__(**kwargs)

        self.data_root_dir = data_root_dir
        self.annotation_dir = annotation_dir
        self.temporal_crops = temporal_crops
        self.spatial_crops = spatial_crops
        self.total_clips = self.temporal_crops * self.spatial_crops
        self.fix_len = fix_len
        self._samples = []
        self._spatial_temporal_index = []

        if _get_samples:
            self._get_samples()

    @abstractmethod
    def _get_samples(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        actual_index = index // self.total_clips
        tmp = index - actual_index * self.total_clips
        if self.mode == "train":
            clip_id = -1
            spatial_id = 0
            num_clips = 1
        else:
            clip_id = tmp // self.spatial_crops
            spatial_id = tmp % self.spatial_crops
            num_clips = self.temporal_crops

        sample_info = self._get(actual_index)
        sample_info["meta"]["clip_id"] = clip_id
        sample_info["meta"]["num_clips"] = num_clips
        crop_list = ["cc", "ll", "rr", "tl", "tr", "bl", "br"]
        crop_mode = crop_list[spatial_id % len(crop_list)]
        sample_info["meta"]["crop_mode"] = crop_mode

        return self.pipeline(sample_info)

    def __len__(self):
        if self.fix_len is not None:
            return self.fix_len * self.total_clips
        return len(self._samples) * self.total_clips

    def _get(self, index: int):
        video_info = self._samples[index]
        ret = {
            "meta": {
                "prefix": self.data_root_dir,
                "video_path": video_info["video_path"]
            },
        }
        if 'gt_label' in video_info:
            ret['gt_label'] = np.array(video_info["gt_label"], dtype=np.int64)
        return ret
