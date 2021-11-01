# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch

from essmc2.utils.metric import accuracy
from .train_module import TrainModule
from ..registry import MODELS, BACKBONES, NECKS, HEADS


@MODELS.register_class()
class Classifier(TrainModule):
    def __init__(self, backbone, neck, head, topk=(1,)):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)
        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = topk

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, img, train_mode=False, **kwargs):
        return self.forward_train(img, **kwargs) if train_mode else self.forward_test(img, **kwargs)

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
