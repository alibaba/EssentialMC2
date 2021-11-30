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
    def __call__(self, results, targets, prefix="acc"):
        """ Compute Accuracy
        Args:
            results (torch.Tensor or numpy.ndarray):
            targets (torch.Tensor or numpy.ndarray):
            prefix (str): Prefix string of ret key, default is acc.

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        assert self.maxk <= results.shape[-1]

        if isinstance(results, np.ndarray):
            results = torch.from_numpy(results)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        batch_size = results.size(0)

        _, pred = results.topk(self.maxk, 1, True, True)
        pred = pred.t()
        corrects = pred.eq(targets.view(1, -1).expand_as(pred))

        res = OrderedDict()
        for k in self.topk:
            correct_k = corrects[:k].contiguous().view(-1).float().sum(0)
            res[f"{prefix}@{k}"] = correct_k.mul_(1.0 / batch_size)
        return res


@METRICS.register_class("accuracyx2")
class AccuracyMetricx2(object):
    def __init__(self, topk=(1,)):
        if isinstance(topk, int):
            topk = (topk,)
        self.topk = topk
        self.maxk = max(self.topk)

    def __call__(self, logits_0, logits_1, targets):
        """ Compute Accuracy
        Args:
            logits_0 (torch.Tensor or numpy.ndarray):
            logits_1 (torch.Tensor or numpy.ndarray)
            targets (torch.Tensor or numpy.ndarray):

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        assert self.maxk <= logits_0.shape[-1]
        assert self.maxk <= logits_1.shape[-1]
        assert targets.shape[-1] == 2

        if isinstance(logits_0, np.ndarray):
            logits_0 = torch.from_numpy(logits_0)
        if isinstance(logits_1, np.ndarray):
            logits_1 = torch.from_numpy(logits_1)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        res = OrderedDict()
        res0 = AccuracyMetric(topk=self.topk)(logits_0, targets[:, 0])
        res1 = AccuracyMetric(topk=self.topk)(logits_1, targets[:, 1])

        for key, value in res0.items():
            key = key.replace("@", "_0@")
            res[key] = value

        for key, value in res1.items():
            key = key.replace("@", "_1@")
            res[key] = value
        return res
