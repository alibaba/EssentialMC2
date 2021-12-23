# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .compose import Compose
from .identity import Identity
from .image import ImageTransform, RandomResizedCrop, RandomHorizontalFlip, Normalize, ImageToTensor
from .io import LoadPILImageFromFile, LoadCvImageFromFile, LoadImageFromFile
from .io_video import DecodeVideoToTensor
from .io_video_new import LoadVideoFromFile
from .registry import TRANSFORMS, build_pipeline
from .tensor import ToTensor, Select
from .video import VideoTransform, RandomResizedCropVideo, CenterCropVideo, \
    RandomHorizontalFlipVideo, NormalizeVideo, VideoToTensor, AutoResizedCropVideo, \
    ResizeVideo
