# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import _pickle as pkl
import os.path as osp
import random

import numpy as np
from PIL import Image

from essmc2.datasets import BaseDataset, DATASETS

ASYM_TRANSITION = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}


@DATASETS.register_class()
class CifarNoisyDataset(BaseDataset):
    def __init__(self,
                 root_dir="",
                 cifar_type="cifar10",
                 noise_mode="sym",
                 noise_ratio=0.5,
                 **kwargs):
        super(CifarNoisyDataset, self).__init__(**kwargs)
        self.root_dir = root_dir
        assert cifar_type in ('cifar10', 'cifar100')
        self.cifar_type = cifar_type
        assert noise_mode in ('asym', 'sym')
        self.noise_mode = noise_mode
        self.noise_ratio = noise_ratio
        self.num_classes = 10 if cifar_type == "cifar10" else 100

        self.images = []
        self.clean_labels = []
        self.noise_labels = []

        self._load_cifar_data()
        self._make_noise()

    def _load_cifar_data(self):
        if self.cifar_type == "cifar10":
            if self.mode == "test":
                pkl_path = osp.join(self.root_dir, 'test_batch')
                with open(pkl_path, "rb") as f:
                    data = pkl.load(f, encoding="latin1")
                    images = data['data']
                    images = images.reshape((10000, 3, 32, 32))
                    images = images.transpose((0, 2, 3, 1))
                    self.images = images
                    self.clean_labels = data['labels']
            else:
                images_list = []
                for i in range(1, 6):
                    pkl_path = osp.join(self.root_dir, f'data_batch_{i}')
                    with open(pkl_path, "rb") as f:
                        data = pkl.load(f, encoding='latin1')
                        images = data['data']
                        images_list.append(images)
                        self.clean_labels.extend(data['labels'])
                images = np.concatenate(images_list)
                images = images.reshape((50000, 3, 32, 32))
                images = images.transpose((0, 2, 3, 1))
                self.images = images
        elif self.cifar_type == "cifar100":
            if self.mode == "test":
                pkl_path = osp.join(self.root_dir, 'test')
            else:
                pkl_path = osp.join(self.root_dir, 'train')
            with open(pkl_path, "rb") as f:
                data = pkl.load(f, encoding='latin1')
                images = data['data']
                images = images.reshape((10000 if self.mode == "test" else 50000, 3, 32, 32))
                images = images.transpose((0, 2, 3, 1))
                self.images = images
                self.clean_labels = data['fine_labels']
        else:
            raise ValueError(f"Unexpected cifar_type, support cifar10, cifar100, got {self.cifar_type}")

    def _make_noise(self):
        self.noise_labels = self.clean_labels.copy()
        if self.mode == "test":
            return

        data_len = len(self.clean_labels)
        idx = list(range(data_len))
        random.shuffle(idx)
        noise_num = int(self.noise_ratio * data_len)
        noise_idx = set(idx[:noise_num])
        for i in range(data_len):
            if i in noise_idx:
                if self.noise_mode == "sym":
                    noise_label = random.randint(0, self.num_classes - 1)
                else:
                    noise_label = ASYM_TRANSITION[self.clean_labels[i]]
                self.noise_labels[i] = noise_label

    def __len__(self) -> int:
        return len(self.images)

    def _get(self, index: int):
        img = Image.fromarray(self.images[index])
        if self.mode == "test":
            gt_label = self.clean_labels[index]
        else:
            gt_label = self.noise_labels[index]
        return dict(img=img,
                    index=np.array(index, dtype=np.int64),
                    gt_label=np.array(gt_label, dtype=np.int64),
                    meta=dict()
                    )
