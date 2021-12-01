# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .compose import compose
from .identity import Identity
from .image import ImageTransform, RandomResizedCrop, RandomHorizontalFlip, Normalize, ImageToTensor
from .io import LoadPILImageFromFile, LoadCvImageFromFile
from .io_video import DecodeVideoToTensor
from .registry import TRANSFORMS, build_pipeline
from .tensor import ToTensor, Select
from .video import VideoTransform, RandomResizedCropVideo, CenterCropVideo, \
    RandomHorizontalFlipVideo, NormalizeVideo, VideoToTensor, AutoResizedCropVideo, \
    ResizeVideo
