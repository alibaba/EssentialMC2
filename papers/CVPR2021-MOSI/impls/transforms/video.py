# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

# ColorJitterVideo class is modified from
# https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py
# MIT License
#
# Copyright (c) 2019 Tengda Han
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numbers
import random

import torch
from torchvision.transforms import Lambda, Compose

from essmc2.transforms import VideoTransform, TRANSFORMS


@TRANSFORMS.register_class()
class ColorJitterVideo(VideoTransform):
    """ Modified from https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py.
    Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale (float): possibility to transform the video to grayscale.
            Should have a value range of [0, 1]
        consistent  (bool): indicates whether or not to keep all the color transformations consistent for all the frames.
        shuffle     (bool): indicates whether or not to shuffle the sequence of the augmentations.
        gray_first  (bool): indicates whether or not to put grayscale transform first.
    """

    def __init__(
            self,
            brightness=0, contrast=0, saturation=0, hue=0, grayscale=0,
            consistent=False, shuffle=True, gray_first=True,
            **kwargs
    ):
        super(ColorJitterVideo, self).__init__(**kwargs)
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

        self.grayscale = grayscale
        self.consistent = consistent
        self.shuffle = shuffle
        self.gray_first = gray_first

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_transform(self, T, device):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Arg:
            T (int): number of frames. Used when consistent = False.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if self.brightness is not None:
            if self.consistent:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            else:
                brightness_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.brightness[0],
                                                                                      self.brightness[1])
            transforms.append(Lambda(lambda frame: adjust_brightness(frame, brightness_factor)))

        if self.contrast is not None:
            if self.consistent:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            else:
                contrast_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda frame: adjust_contrast(frame, contrast_factor)))

        if self.saturation is not None:
            if self.consistent:
                saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            else:
                saturation_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.saturation[0],
                                                                                      self.saturation[1])
            transforms.append(Lambda(lambda frame: adjust_saturation(frame, saturation_factor)))

        if self.hue is not None:
            if self.consistent:
                hue_factor = random.uniform(self.hue[0], self.hue[1])
            else:
                hue_factor = torch.empty([T, 1, 1], device=device).uniform_(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda frame: adjust_hue(frame, hue_factor)))

        if self.shuffle:
            random.shuffle(transforms)

        if random.uniform(0, 1) < self.grayscale:
            gray_transform = Lambda(lambda frame: rgb_to_grayscale(frame))
            if self.gray_first:
                transforms.insert(0, gray_transform)
            else:
                transforms.append(gray_transform)

        transform = Compose(transforms)

        return transform

    def __call__(self, item):
        clip = item[self.input_key]
        raw_shape = clip.shape  # (C, T, H, W)
        device = clip.device
        T = raw_shape[1]
        transform = self._get_transform(T, device)
        clip = transform(clip)
        assert clip.shape == raw_shape
        item[self.output_key] = clip
        return item

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string


def _is_tensor_a_torch_image(input):
    return input.ndim >= 2


def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
        Args:
            clip (torch.tensor): Size is (T, H, W, C)
        Return:
            clip (torch.tensor): Size is (T, H, W, C)
    """
    orig_dtype = img.dtype
    rgb_convert = torch.tensor([0.299, 0.587, 0.114])

    assert img.shape[0] == 3, "First dimension need to be 3 Channels"
    if img.is_cuda:
        rgb_convert = rgb_convert.to(img.device)

    img = img.float().permute(1, 2, 3, 0).matmul(rgb_convert).to(orig_dtype)
    return torch.stack([img, img, img], 0)


def _rgb2hsv(img):
    r, g, b = img.unbind(0)

    maxc, _ = torch.max(img, dim=0)
    minc, _ = torch.min(img, dim=0)

    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))


def _hsv2rgb(img):
    l = len(img.shape)
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    if l == 3:
        tmp = torch.arange(6)[:, None, None]
    elif l == 4:
        tmp = torch.arange(6)[:, None, None, None]

    if img.is_cuda:
        tmp = tmp.to(img.device)

    mask = i == tmp  # (H, W) == (6, H, W)

    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3))  # (3, 6, H, W)

    if l == 3:
        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4)  # (C, H, W)
    elif l == 4:
        return torch.einsum("itjk, xitjk -> xtjk", mask.to(dtype=img.dtype), a4)  # (C, T, H, W)


def adjust_brightness(img, brightness_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img, contrast_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(rgb_to_grayscale(img).to(torch.float), dim=(-4, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_saturation(img, saturation_factor):
    # type: (Tensor, float) -> Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    if isinstance(hue_factor, float) and not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    elif isinstance(hue_factor, torch.Tensor) and not (
            (-0.5 <= hue_factor).sum() == hue_factor.shape[0] and (hue_factor <= 0.5).sum() == hue_factor.shape[0]):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(0)
    h += hue_factor
    h = h % 1.0
    img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj
