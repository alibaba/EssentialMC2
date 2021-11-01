# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from essmc2.models.registry import HEADS


@HEADS.register_class()
class NoisyContrastHead(nn.Module):
    def __init__(self, in_channels, num_classes, out_feat_dim):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_classes)
        self.feat_extractor = nn.Linear(in_channels, out_feat_dim)
        self.reconstructor = nn.Linear(out_feat_dim, in_channels)
        self.num_classes = num_classes
        self.out_feat_dim = out_feat_dim

    def forward(self, x, do_classify=True, do_extract_feature=True, do_reconstruct=False):
        if do_classify:
            logits = self.classifier(x)
        else:
            logits = None
        if do_extract_feature:
            features = self.feat_extractor(x)
            features_norm = F.normalize(features)
        else:
            features, features_norm = None, None
        if do_reconstruct:
            if features is None:
                features = self.feat_extractor(x)
            recons = F.relu(self.reconstructor(features))
            error = torch.mean((recons - features) ** 2, dim=1)
        else:
            recons, error = None, None

        if do_classify:
            if do_extract_feature and do_reconstruct:
                return logits, features_norm, error
            elif do_extract_feature:
                return logits, features_norm
            elif do_reconstruct:
                return logits, error
            else:
                return logits
        else:
            if do_extract_feature and do_reconstruct:
                return features_norm, error
            elif do_extract_feature:
                return features_norm
            elif do_reconstruct:
                return error
            else:
                return None
