# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os

import matplotlib.pyplot as plt
import torch.nn as nn


class Visualize3DModule(nn.Module):
    def __init__(self, visualize=False, visualize_output_dir=""):
        super(Visualize3DModule, self).__init__()
        self.visualize = visualize
        self.visualize_output_dir = visualize_output_dir
        self.id = 0

    def visualize_features(self, module, input_x, output_x):
        """
        Visualizes and saves the normalized output features for the module.
        """
        if not self.visualize:
            return
        b, c, t, h, w = output_x.shape
        xmin, xmax = output_x.min(1).values.unsqueeze(1), output_x.max(1).values.unsqueeze(1)
        x_vis = ((output_x.detach() - xmin) / (xmax - xmin)).permute(0, 1, 3, 2, 4) \
            .reshape(b, c * h, t * w).detach().cpu().numpy()
        if hasattr(self, "stage_id"):
            stage_id = self.stage_id
            block_id = self.block_id
        else:
            stage_id = 0
            block_id = 0
        for i in range(b):
            if not os.path.exists(f'{self.visualize_output_dir}/im_{self.id + i}/'):
                os.makedirs(f'{self.visualize_output_dir}/im_{self.id + i}/')
            plt.imsave(
                f'{self.base_output_dir}/'
                f'im_{self.id + i}/layer_{stage_id}_{block_id}_feature.jpg',
                x_vis[i]
            )
        self.id += b

    def set_stage_block_id(self, stage_id, block_id):
        setattr(self, "stage_id", stage_id)
        setattr(self, "block_id", block_id)
