# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numpy as np
import torch
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_cv2_image(img):
    return isinstance(img, np.ndarray) and img.dtype == np.uint8


def is_tensor(t):
    return isinstance(t, torch.Tensor)


INPUT_PIL_TYPE_WARNING = 'input should be PIL Image'
INPUT_CV2_TYPE_WARNING = 'input should be cv2 image(uint8 np.ndarray)'
INPUT_TENSOR_TYPE_WARNING = 'input should be tensor(uint8 np.ndarray)'
