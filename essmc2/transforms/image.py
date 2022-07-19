# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numpy as np
import opencv_transforms.transforms as cv2_transforms
import torch
import torchvision.transforms as transforms

from .registry import TRANSFORMS
from .utils import is_pil_image, INPUT_PIL_TYPE_WARNING, \
    is_cv2_image, INPUT_CV2_TYPE_WARNING, is_tensor, INPUT_TENSOR_TYPE_WARNING, \
    BACKEND_TORCHVISION, BACKEND_PILLOW, BACKEND_CV2, TORCHVISION_CAPABILITY, \
    INTERPOLATION_STYLE, INTERPOLATION_STYLE_CV2

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
    """ Crop a random portion of image.
    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (sequence or int): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
        padding (sequence or int): Optional padding on each border of the image. Default is None.
        pad_if_needed (bool): It will pad the image if smaller than the desired size to avoid raising an exception.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
    """

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


@TRANSFORMS.register_class()
class RandomResizedCrop(ImageTransform):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (int or sequence): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear', **kwargs):
        super(RandomResizedCrop, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.interpolation = interpolation
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert interpolation in INTERPOLATION_STYLE
        else:
            assert interpolation in INTERPOLATION_STYLE_CV2
        self.callable = transforms.RandomResizedCrop(size, scale, ratio, INTERPOLATION_STYLE[interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) \
            else cv2_transforms.RandomResizedCrop(size, scale, ratio, INTERPOLATION_STYLE_CV2[interpolation])

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class Resize(ImageTransform):
    """Resize image to a given size.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (int or sequence): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """

    def __init__(self, size, interpolation='bilinear', **kwargs):
        super(Resize, self).__init__(**kwargs)
        assert self.backend in BACKENDS
        self.size = size
        self.interpolation = interpolation
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert interpolation in INTERPOLATION_STYLE
        else:
            assert interpolation in INTERPOLATION_STYLE_CV2
        self.callable = transforms.Resize(size, INTERPOLATION_STYLE[interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) \
            else cv2_transforms.Resize(size, INTERPOLATION_STYLE_CV2[interpolation])

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
        return item


@TRANSFORMS.register_class()
class CenterCrop(ImageTransform):
    """ Crops the given image at the center.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (sequence or int): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
    """

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
    """ Horizontally flip the given image randomly with a given probability.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

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
    """ Normalize a tensor image with mean and standard deviation.
    This transform only support tensor image.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

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
    """ Convert a ``PIL Image`` or ``numpy.ndarray`` or uint8 type tensor to a float32 tensor,
    and scale output to [0.0, 1.0].
    """

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
