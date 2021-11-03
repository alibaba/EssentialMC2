# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import _pickle as pkl
import os.path as osp
import random

import numpy as np

from essmc2.datasets import DATASETS
from .cifar_noisy_dataset import CifarNoisyDataset


@DATASETS.register_class()
class CifarNoisyOpensetDataset(CifarNoisyDataset):
    def __init__(self,
                 root_dir="",
                 cifar_type="cifar10",
                 noise_mode="sym",
                 noise_ratio=0.5,
                 ood_noise_name="place365",
                 ood_noise_root_dir="",
                 ood_noise_num=20000,
                 train_used_idx=None,
                 **kwargs):
        super(CifarNoisyOpensetDataset, self).__init__(root_dir, cifar_type, noise_mode, noise_ratio, **kwargs)
        self.ood_noise_name = ood_noise_name
        assert self.ood_noise_name in ("place365", "tiny_imagenet", "cifar100")
        if self.cifar_type == "cifar100" and self.ood_noise_name == "cifar100":
            raise Exception(f"Original input and ood noise input are same {self.cifar_type}")
        self.ood_noise_root_dir = ood_noise_root_dir
        self.ood_noise_num = ood_noise_num
        self.train_used_idx = train_used_idx or []

        self.openset_select_ids = []
        self.openset_images = []
        self.openset_noise_labels = []
        self.openset_clean_labels = []

        self._load_openset_noise(train_used_idx)

    def _load_openset_noise(self, train_used_noise: list):
        if self.ood_noise_name == "place365":
            openset_images = np.load(osp.join(self.ood_noise_root_dir, "places365_test.npy")).transpose((0, 2, 3, 1))
        elif self.ood_noise_name == "tiny_imagenet":
            openset_images = np.load(osp.join(self.ood_noise_root_dir, "TIN_train.npy")).transpose((0, 2, 3, 1))
        else:
            # cifar100
            with open(osp.join(self.ood_noise_root_dir, 'train'), "rb") as f:
                openset_images = pkl.load(f, encoding='latin1')['data'].reshape((50000, 3, 32, 32)).transpose(
                    (0, 2, 3, 1))

        idx = list(range(openset_images.shape[0]))
        if self.mode == "test":
            idx = list(set(idx) - set(train_used_noise))
        random.shuffle(idx)
        self.ood_noise_name = min(self.ood_noise_num, len(idx))
        idx = idx[0: self.ood_noise_num]
        self.openset_select_ids = idx

        self.openset_noise_labels = np.random.choice(list(range(self.num_classes)),
                                                     size=self.ood_noise_num,
                                                     replace=True).tolist()
        self.openset_clean_labels = [-1] * self.ood_noise_num
        self.openset_images = openset_images[idx]

        self.images = np.concatenate([self.images, self.openset_images])
        self.noise_labels = self.noise_labels + self.openset_noise_labels
        self.clean_labels = self.clean_labels + self.openset_clean_labels
