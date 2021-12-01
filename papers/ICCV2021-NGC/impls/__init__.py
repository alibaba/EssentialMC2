# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .augmix import AugMix
from .backbones import preresnet
from .datasets.cifar_noisy_dataset import CifarNoisyDataset
from .datasets.cifar_noisy_openset_dataset import CifarNoisyOpensetDataset
from .datasets.imagenet import ImageNet
from .datasets.webvision import Webvision
from .heads import NoisyContrastHead
from .networks import NGCNetwork
from .solvers import NGCSolver

DOC = "ICCV2021-NGC implementations"

__all__ = [
    "NoisyContrastHead", "NGCNetwork",
    "preresnet", "NGCSolver",
    "DOC",
    "CifarNoisyDataset", "CifarNoisyOpensetDataset",
    "Webvision", "ImageNet",
    "AugMix"
]
