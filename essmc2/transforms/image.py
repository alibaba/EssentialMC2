# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import cv2
import numpy as np
import opencv_transforms.transforms as cv2_transforms
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.version import __version__ as tv_version
from packaging.version import parse as parse_version

from .registry import TRANSFORMS
from .utils import is_pil_image, INPUT_PIL_TYPE_WARNING, \
    is_cv2_image, INPUT_CV2_TYPE_WARNING, is_tensor, INPUT_TENSOR_TYPE_WARNING

BACKEND_PILLOW = "pillow"
BACKEND_CV2 = "cv2"
BACKEND_TORCHVISION = "torchvision"
TORCHVISION_CAPABILITY = parse_version(tv_version) >= parse_version('0.8.0')
if TORCHVISION_CAPABILITY:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2, BACKEND_TORCHVISION)
else:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2)


class ImageTransform(object):
    def __init__(self, backend=BACKEND_PILLOW, input_key=None, output_key=None):
        self.input_key = input_key or "img"
        self.output_key = output_key or "img"
        self.backend = backend

    def check_image_type(self, input_img):
        if self.backend == BACKEND_PILLOW:
            assert is_pil_image(input_img), INPUT_PIL_TYPE_WARNING
        elif self.backend == BACKEND_CV2:
            assert is_cv2_image(input_img), INPUT_CV2_TYPE_WARNING
        elif TORCHVISION_CAPABILITY:
            if self.backend == BACKEND_TORCHVISION:
                assert is_tensor(input_img), INPUT_TENSOR_TYPE_WARNING


@TRANSFORMS.register_class()
class RandomCrop(ImageTransform):
    def __init__(self, size,
                 padding=None, pad_if_needed=False, fill=0, padding_mode='constant', **kwargs):
        super(RandomCrop, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            self.callable = transforms.RandomCrop(size,
                                                  padding=padding, pad_if_needed=pad_if_needed, fill=fill,
                                                  padding_mode=padding_mode)
        else:
            self.callable = cv2_transforms.RandomCrop(size,
                                                      padding=padding, pad_if_needed=pad_if_needed, fill=fill,
                                                      padding_mode=padding_mode)

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


interpolation_style = {
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "bicubic": Image.BICUBIC,
}
interpolation_style_cv2 = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
}


@TRANSFORMS.register_class()
class RandomResizedCrop(ImageTransform):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear', **kwargs):
        super(RandomResizedCrop, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.interpolation = interpolation
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert interpolation in interpolation_style
        else:
            assert interpolation in interpolation_style_cv2
        self.callable = transforms.RandomResizedCrop(size, scale, ratio, interpolation_style[interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.RandomResizedCrop(size, scale,
                                                                                                           ratio,
                                                                                                           interpolation_style_cv2[
                                                                                                               interpolation])

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class Resize(ImageTransform):
    def __init__(self, size, interpolation='bilinear', **kwargs):
        super(Resize, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.size = size
        self.interpolation = interpolation
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert interpolation in interpolation_style
        else:
            assert interpolation in interpolation_style_cv2
        self.callable = transforms.Resize(size, interpolation_style[interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.Resize(size,
                                                                                                interpolation_style_cv2[
                                                                                                    interpolation])

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class CenterCrop(ImageTransform):
    def __init__(self, size, **kwargs):
        super(CenterCrop, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.size = size
        self.callable = transforms.CenterCrop(size) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.CenterCrop(size)

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class RandomHorizontalFlip(ImageTransform):
    def __init__(self, p=0.5, **kwargs):
        super(RandomHorizontalFlip, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.callable = transforms.RandomHorizontalFlip(p) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.RandomHorizontalFlip(p)

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class Normalize(ImageTransform):
    def __init__(self, mean, std, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.callable = transforms.Normalize(self.mean, self.std) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.Normalize(self.mean, self.std)

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        # item["meta"]["normalize_params"] = dict(mean=self.mean, std=self.std)
        return item


@TRANSFORMS.register_class()
class ImageToTensor(ImageTransform):
    def __init__(self, **kwargs):
        super(ImageToTensor, self).__init__(**kwargs)
        assert self.backend in BACKENDS

        if self.backend == BACKEND_PILLOW:
            self.callable = transforms.ToTensor()
        elif self.backend == BACKEND_CV2:
            self.callable = cv2_transforms.ToTensor()
        else:
            self.callable = transforms.ConvertImageDtype(torch.float)

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        return item
