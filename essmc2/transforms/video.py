# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import random

import numpy as np
import torchvision.transforms.functional as functional
import torchvision.transforms.transforms as transforms
from packaging import version
from torchvision.version import __version__ as tv_version

from .registry import TRANSFORMS
from .utils import is_tensor, BACKEND_TORCHVISION, INTERPOLATION_STYLE

# torchvision.transforms._transforms_video is deprecated since torchvision 0.10.0, use transforms instead
use_video_transforms = version.parse(tv_version) < version.parse("0.10.0")

BACKENDS = (BACKEND_TORCHVISION,)


class VideoTransform(object):
    def __init__(self, backend=BACKEND_TORCHVISION, input_key=None, output_key=None):
        self.input_key = input_key or "video"
        self.output_key = output_key or "video"
        self.backend = backend

    def check_video_type(self, input_video):
        if self.backend == BACKEND_TORCHVISION:
            assert is_tensor(input_video)


@TRANSFORMS.register_class()
class RandomResizedCropVideo(VideoTransform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear', **kwargs):
        super(RandomResizedCropVideo, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.interpolation = interpolation

        if use_video_transforms:
            from torchvision.transforms._transforms_video import RandomResizedCropVideo as RandomResizedCropVideoOp
            self.callable = RandomResizedCropVideoOp(size, scale, ratio, self.interpolation)
        else:
            self.callable = transforms.RandomResizedCrop(size, scale, ratio, INTERPOLATION_STYLE[self.interpolation])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class CenterCropVideo(VideoTransform):
    def __init__(self, size, **kwargs):
        super(CenterCropVideo, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.size = size

        if use_video_transforms:
            from torchvision.transforms._transforms_video import CenterCropVideo as CenterCropVideoOp
            self.callable = CenterCropVideoOp(size)
        else:
            self.callable = transforms.CenterCrop(size)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class RandomHorizontalFlipVideo(VideoTransform):
    def __init__(self, p=0.5, **kwargs):
        super(RandomHorizontalFlipVideo, self).__init__(**kwargs)
        assert self.backend in BACKENDS

        if use_video_transforms:
            from torchvision.transforms._transforms_video import \
                RandomHorizontalFlipVideo as RandomHorizontalFlipVideoOp
            self.callable = RandomHorizontalFlipVideoOp(p)
        else:
            self.callable = transforms.RandomHorizontalFlip(p)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class NormalizeVideo(VideoTransform):
    def __init__(self, mean, std, **kwargs):
        super(NormalizeVideo, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if use_video_transforms:
            from torchvision.transforms._transforms_video import NormalizeVideo as NormalizeVideoOp
            self.callable = NormalizeVideoOp(self.mean, self.std)
        else:
            self.callable = transforms.Normalize(self.mean, self.std)

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class VideoToTensor(VideoTransform):
    def __init__(self, **kwargs):
        super(VideoToTensor, self).__init__(**kwargs)
        assert self.backend in BACKENDS

        if use_video_transforms:
            from torchvision.transforms._transforms_video import ToTensorVideo
            self.callable = ToTensorVideo()
        else:
            self.callable = transforms.ToTensor()

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class AutoResizedCropVideo(VideoTransform):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 interpolation_mode='bilinear',
                 **kwargs):
        super(AutoResizedCropVideo, self).__init__(**kwargs)
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale
        self.interpolation_mode = interpolation_mode

    def get_crop(self, clip, crop_mode='cc'):
        scale = random.uniform(*self.scale)
        _, _, video_height, video_width = clip.shape
        min_length = min(video_height, video_width)
        crop_size = int(min_length * scale)
        center_x = video_width // 2
        center_y = video_height // 2
        box_half = crop_size // 2

        # default is cc
        x0 = center_x - box_half
        y0 = center_y - box_half
        if crop_mode == "ll":
            x0 = 0
            y0 = center_y - box_half
        elif crop_mode == "rr":
            x0 = video_width - crop_size
            y0 = center_y - box_half
        elif crop_mode == "tl":
            x0 = 0
            y0 = 0
        elif crop_mode == "tr":
            x0 = video_width - crop_size
            y0 = 0
        elif crop_mode == "bl":
            x0 = 0
            y0 = video_height - crop_size
        elif crop_mode == "br":
            x0 = video_width - crop_size
            y0 = video_height - crop_size

        if use_video_transforms:
            from torchvision.transforms._functional_video import resized_crop
            return resized_crop(clip, y0, x0, crop_size, crop_size, self.size, self.interpolation_mode)
        else:
            return functional.resized_crop(clip, y0, x0, crop_size, crop_size, self.size,
                                           INTERPOLATION_STYLE[self.interpolation_mode])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        crop_mode = item["meta"].get("crop_mode") or "cc"
        item[self.output_key] = self.get_crop(item[self.input_key], crop_mode)
        return item


@TRANSFORMS.register_class()
class ResizeVideo(VideoTransform):
    def __init__(self,
                 size,
                 interpolation_mode='bilinear',
                 **kwargs):
        super(ResizeVideo, self).__init__(**kwargs)
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode

    def resize(self, clip):
        if use_video_transforms:
            from torchvision.transforms._functional_video import resize
            return resize(clip, self.size, self.interpolation_mode)
        else:
            return functional.resize(clip, self.size, INTERPOLATION_STYLE[self.interpolation_mode])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.resize(item[self.input_key])
        return item
