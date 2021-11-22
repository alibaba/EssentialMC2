# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

from essmc2.models.networks.train_module import TrainModule
from essmc2.models.registry import MODELS, BACKBONES, NECKS, HEADS, LOSSES
from essmc2.utils.logger import get_logger
from essmc2.utils.metric import accuracy
from essmc2.utils.model import load_pretrained


@MODELS.register_class()
class Classifier(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 loss=None,
                 topk=(1,)):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)

        self.loss = LOSSES.build(loss or dict(type='CrossEntropy'))

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = topk

    def forward(self, img, **kwargs):
        return self.forward_train(img, **kwargs) if self.training else self.forward_test(img, **kwargs)

    def forward_train(self, img, gt_label, **kwargs):
        ret = OrderedDict()
        logits = self.head(self.neck(self.backbone(img)))
        loss = self.loss(logits, gt_label)
        ret["loss"] = loss
        acc = accuracy(logits, gt_label, topk=self.topk)
        for k, acc_at_k in zip(self.topk, acc):
            ret[f"acc@{k}"] = acc_at_k
        return ret

    def forward_test(self, img, gt_label=None, **kwargs):
        logits = torch.nn.functional.softmax(self.head(self.neck(self.backbone(img))), dim=1)
        if gt_label is None:
            return logits
        ret = OrderedDict()
        acc = accuracy(logits, gt_label, topk=self.topk)
        for k, acc_at_k in zip(self.topk, acc):
            ret[f"acc@{k}"] = acc_at_k
        return ret


@MODELS.register_class()
class VideoClassifier(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 loss=None,
                 freeze_bn=False,
                 topk=(1,),
                 use_pretrain=False,
                 load_from=""):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        if neck is not None:
            self.neck = NECKS.build(neck)
        else:
            self.neck = None
        self.head = HEADS.build(head)

        self.loss = LOSSES.build(loss or dict(type='CrossEntropy'))

        self.freeze_bn = freeze_bn

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = topk

        self.loss = torch.nn.CrossEntropyLoss()

        if use_pretrain:
            load_pretrained(self, load_from, logger=get_logger())

    def train(self, mode=True):
        self.training = mode
        super(VideoClassifier, self).train(mode=mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.freeze_bn:
                module.train(False)
        return self

    def forward(self, video, **kwargs):
        return self.forward_train(video, **kwargs) if self.training else self.forward_test(video, **kwargs)

    def forward_train(self, video, gt_label, **kwargs):
        ret = OrderedDict()

        x = self.backbone(video)
        if self.neck is not None:
            x = self.neck(x)
        probs = self.head(x)

        if gt_label is None:
            return probs

        loss = self.loss(probs, gt_label)
        ret["loss"] = loss
        with torch.no_grad():
            acc_topk = accuracy(probs, gt_label, self.topk)
        for acc, k in zip(acc_topk, self.topk):
            ret[f"acc@{k}"] = acc
        ret["batch_size"] = video.size(0)
        return ret

    def forward_test(self, video, gt_label=None, **kwargs):
        x = self.backbone(video)
        if self.neck is not None:
            x = self.neck(x)
        probs = self.head(x)

        if type(probs) is tuple:
            probs = tuple([nn.functional.softmax(t, dim=1) for t in probs])
        else:
            probs = nn.functional.softmax(probs, dim=1)

        if gt_label is None:
            return probs
        ret = OrderedDict()
        acc_topk = accuracy(probs, gt_label, self.topk)
        for acc, k in zip(acc_topk, self.topk):
            ret[f"acc@{k}"] = acc
        ret["batch_size"] = video.size(0)
        return ret
