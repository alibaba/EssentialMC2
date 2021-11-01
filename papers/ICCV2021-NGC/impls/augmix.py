# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import numpy as np
import torch
import torchvision.transforms as transforms

from essmc2.transforms import ImageTransform, TRANSFORMS


@TRANSFORMS.register_class()
class AugMix(ImageTransform):
    def __init__(self, mixture_width=3, mixture_depth=-1, aug_prob_coeff=1.0, aug_severity=1,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        super(AugMix, self).__init__(**kwargs)
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_prob_coeff = aug_prob_coeff
        self.aug_severity = aug_severity
        self.mean = mean
        self.std = std
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        assert self.backend == "pillow", "AugMix only support pillow backend"

    def __call__(self, item):
        self.check_image_type(item[self.input_key])
        item[self.output_key] = AugMix.aug(item[self.input_key], self.preprocess,
                                           self.mixture_width, self.mixture_depth, self.aug_prob_coeff,
                                           self.aug_severity)
        return item

    @staticmethod
    def aug(image, preprocess, mixture_width=3, mixture_depth=-1, aug_prob_coeff=-1.0, aug_severity=1):
        from .augmix_impls import augmentations_all

        ws = np.float32(
            np.random.dirichlet([aug_prob_coeff] * mixture_width))
        m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

        preprocessed = preprocess(image)
        mix = torch.zeros_like(preprocessed)
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations_all)
            image_aug = op(image_aug, aug_severity)
        mix += ws[i] * preprocess(image_aug)

        mixed = (1.0 - m) * preprocessed + m * mix
        return mixed
