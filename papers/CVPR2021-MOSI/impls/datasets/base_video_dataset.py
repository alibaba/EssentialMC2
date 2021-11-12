# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp
from abc import abstractmethod, ABCMeta

from essmc2.datasets import BaseDataset
from essmc2.utils.file_systems import FS


class BaseVideoDataset(BaseDataset, metaclass=ABCMeta):
    """ BaseVideoDataset Interface.

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
                 **kwargs):
        super(BaseVideoDataset, self).__init__(**kwargs)

        self.data_root_dir = data_root_dir
        self.annotation_dir = annotation_dir
        self.temporal_crops = temporal_crops
        self.spatial_crops = spatial_crops
        self.num_clips = self.temporal_crops * self.spatial_crops
        self.fix_len = fix_len
        self._samples = []
        self._spatial_temporal_index = []

        self._construct_dataset()

    def _construct_dataset(self):
        dataset_list_name = self._get_dataset_list_name()

        file_path = osp.join(self.annotation_dir, dataset_list_name)

        with FS.get_fs_client(file_path) as client:
            local_file = client.get_object_to_local_file(file_path)
            if local_file[-4:] == ".csv":
                import pandas
                lines = pandas.read_csv(local_file)
                for line in lines.values.tolist():
                    self._samples.append(line)
            elif local_file[-4:] == "json":
                import json
                with open(local_file, "r") as f:
                    lines = json.load(f)
                for line in lines:
                    self._samples.append(line)
            else:
                with open(local_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        self._samples.append(line.strip())

    @abstractmethod
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset.
        Returns:
            name (str): name of the list to be read
        """
        raise NotImplementedError

    def __getitem__(self, index: int):
        # For each new worker, init file system now.
        self._mp_init_fs()

        actual_index = index // self.num_clips
        tmp = index - actual_index * self.num_clips
        if self.mode == "train":
            clip_id = -1
            spatial_id = 0
            num_clips = 1
        else:
            clip_id = tmp // self.spatial_crops
            spatial_id = tmp % self.spatial_crops
            num_clips = self.temporal_crops

        sample_info = self._get(index)
        sample_info["meta"]["clip_id"] = clip_id
        sample_info["meta"]["num_clips"] = num_clips
        crop_list = ["cc", "ll", "rr", "tl", "tr", "bl", "br"]
        crop_mode = crop_list[spatial_id % len(crop_list)]
        sample_info["meta"]["crop_mode"] = crop_mode

        return self.pipeline(sample_info)

    def __len__(self):
        if self.fix_len is not None:
            return self.fix_len * self.num_clips
        return len(self._samples) * self.num_clips
