# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import random

import numpy as np
import torchvision.transforms._functional_video as F
import torchvision.transforms._transforms_video as transforms_videos

from .registry import TRANSFORMS
from .utils import is_tensor

BACKEND_TORCHVISION = "torchvision"
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

        self.callable = transforms_videos.RandomResizedCropVideo(size, scale, ratio, self.interpolation)

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
        self.callable = transforms_videos.CenterCropVideo(size)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class RandomHorizontalFlipVideo(VideoTransform):
    def __init__(self, p=0.5, **kwargs):
        super(RandomHorizontalFlipVideo, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.callable = transforms_videos.RandomHorizontalFlipVideo(p)

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
        self.callable = transforms_videos.NormalizeVideo(self.mean, self.std)

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class VideoToTensor(VideoTransform):
    def __init__(self, **kwargs):
        super(VideoToTensor, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.callable = transforms_videos.ToTensorVideo()

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

        return F.resized_crop(clip, y0, x0, crop_size, crop_size, self.size, self.interpolation_mode)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        crop_mode = item["meta"].get("crop_mode") or "cc"
        item[self.output_key] = self.get_crop(item[self.input_key], crop_mode)
        return item
