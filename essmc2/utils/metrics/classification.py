# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import numpy as np
import torch

from .registry import METRICS


@METRICS.register_class("accuracy")
class AccuracyMetric(object):
    def __init__(self, topk=(1,)):
        if isinstance(topk, int):
            topk = (topk,)
        self.topk = topk
        self.maxk = max(self.topk)

    @torch.no_grad()
    def __call__(self, logits, labels, prefix="acc"):
        """ Compute Accuracy
        Args:
            logits (torch.Tensor or numpy.ndarray):
            labels (torch.Tensor or numpy.ndarray):
            prefix (str): Prefix string of ret key, default is acc.

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        num_classes = logits.shape[-1]
        maxk = min(self.maxk, num_classes)

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        batch_size = logits.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        corrects = pred.eq(labels.view(1, -1).expand_as(pred))

        res = OrderedDict()
        for k in self.topk:
            if k > maxk:
                continue
            correct_k = corrects[:k].contiguous().view(-1).float().sum(0)
            res[f"{prefix}@{k}"] = correct_k.mul_(1.0 / batch_size)
        return res


@METRICS.register_class("ensemble_accuracy")
class EnsembleAccuracyMetric(object):
    def __init__(self, topk=(1,), ensemble_method="avg"):
        if isinstance(topk, int):
            topk = (topk,)
        self.topk = topk
        self.maxk = max(self.topk)
        assert ensemble_method in ('avg', 'max'), f"Expected ensemble_method in ('avg', 'max'), got {ensemble_method}"
        self.ensemble_method = ensemble_method

    @torch.no_grad()
    def __call__(self, logits, labels, keys, prefix="acc"):
        """ Compute Accuracy
        Args:
            logits (torch.Tensor or numpy.ndarray):
            labels (torch.Tensor or numpy.ndarray):
            keys (List[str]): Keys to accumulate logits.
            prefix (str): Prefix string of ret key, default is acc.

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        agg_keys = list(set(keys))
        keys = np.asarray(keys)

        agg_logits = []  # N * Tensor([C])
        agg_labels = []  # N * Tensor(scalar)

        for key in agg_keys:
            key_index = np.where(keys == key)[0]
            key_index = torch.from_numpy(key_index)
            key_logits = logits[key_index]

            if self.ensemble_method == "avg":
                key_logit = torch.mean(key_logits, dim=0)
            else:
                key_logit, _ = torch.max(key_logit, dim=0)
            key_label = labels[key_index[0]]

            agg_logits.append(key_logit)
            agg_labels.append(key_label)

        agg_logits = torch.vstack(agg_logits)
        agg_labels = torch.hstack(agg_labels)

        return AccuracyMetric(self.topk)(agg_logits, agg_labels, prefix)
