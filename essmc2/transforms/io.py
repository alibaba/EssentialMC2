# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os

import cv2
import numpy as np
import torch
from PIL import Image

from .registry import TRANSFORMS
from ..utils.file_systems import FS


@TRANSFORMS.register_class()
class LoadImageFromFile(object):
    """ Load Image from file. We have multi ways to load image. Here we compose them into one transform.

    Args:
        rgb_order (str): 'RGB' or 'BGR'.
        backend (str): 'pillow', 'cv2' or 'torchvision'. Image should be read as uint8 dtype.
            - 'pillow': Read image file as PIL.Image object.
            - 'cv2': Read image file as numpy.ndarray object.
            - 'torchvision': Read image file as tensor object.
    """

    def __init__(self, rgb_order='RGB', backend='pillow'):
        assert rgb_order in ('RGB', 'BGR')
        assert backend in ('pillow', 'cv2', 'torchvision')
        self.rgb_order = rgb_order
        self.backend = backend

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'], item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']

        with FS.get_from(img_path) as img_path:
            if self.backend == "pillow":
                image = Image.open(img_path).convert(self.rgb_order)
            elif self.backend == "cv2":
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if self.rgb_order == 'RGB':
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            else:
                image = Image.open(img_path).convert(self.rgb_order)
                image_np = np.asarray(image).transpose((2, 0, 1))  # Tensor type needs shape to be (C, H, W)
                image = torch.from_numpy(image_np)

        item['img'] = image
        item['meta']['rgb_order'] = self.rgb_order
        return item


@TRANSFORMS.register_class()
class LoadPILImageFromFile(object):
    def __init__(self, rgb_order='RGB'):
        assert rgb_order in ('RGB', 'BGR')
        self.rgb_order = rgb_order

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'], item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']

        with FS.get_from(img_path) as img_path:
            image = Image.open(img_path).convert(self.rgb_order)
            item['img'] = image
            item['meta']['rgb_order'] = self.rgb_order
        return item


@TRANSFORMS.register_class()
class LoadCvImageFromFile(object):
    def __init__(self, rgb_order='RGB'):
        assert rgb_order in ('RGB', 'BGR')
        self.rgb_order = rgb_order

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'], item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']

        with FS.get_from(img_path) as img_path:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.rgb_order == 'RGB':
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        item['img'] = image
        item['meta']['rgb_order'] = self.rgb_order
        return item
