# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch.nn as nn

from essmc2.models.registry import STEMS
from ..visualize_3d_module import Visualize3DModule


@STEMS.register_class()
class PatchEmbedStem(Visualize3DModule):
    def __init__(self,
                 image_size=224,
                 num_input_channels=3,
                 num_frames=16,
                 patch_size=16,
                 dim=768,
                 **kwargs):
        super(PatchEmbedStem, self).__init__(**kwargs)
        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels=num_input_channels,
            out_channels=dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x):
        b, c, t, h, w, p = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x


@STEMS.register_class()
class TubeletEmbeddingStem(Visualize3DModule):
    def __init__(self,
                 image_size=224,
                 num_input_channels=3,
                 num_frames=16,
                 patch_size=16,
                 dim=768,
                 tubelet_size=2,
                 **kwargs):
        super().__init__(**kwargs)

        num_patches_per_image = (image_size // patch_size) ** 2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(
            in_channels=num_input_channels,
            out_channels=dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        b, c, t, h, w, p = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4], self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x
