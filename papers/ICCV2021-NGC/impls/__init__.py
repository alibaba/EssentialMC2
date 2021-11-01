# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .heads import NoisyContrastHead
from .networks import NGCNetwork
from .backbones import preresnet
from .solvers import NGCSolver
from .datasets.cifar_noisy_dataset import CifarNoisyDataset
from .datasets.cifar_noisy_openset_dataset import CifarNoisyOpensetDataset
from .datasets.webvision import Webvision
from .datasets.imagenet import ImageNet
from .augmix import AugMix

DOC = "ICCV2021-NGC implementations"

__all__ = [
    "NoisyContrastHead", "NGCNetwork",
    "preresnet", "NGCSolver",
    "DOC",
    "CifarNoisyDataset", "CifarNoisyOpensetDataset",
    "Webvision", "ImageNet",
    "AugMix"
]
