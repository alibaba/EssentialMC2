# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

from essmc2.models import TrainModule, MODELS, BACKBONES, NECKS, HEADS
from essmc2.utils.logger import get_logger
from essmc2.utils.metric import accuracy
from essmc2.utils.model import load_pretrained


@MODELS.register_class()
class VideoClassifier(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 freeze_bn=False,
                 topk=(1,),
                 use_pretrain=False,
                 load_from=""):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)
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

    def forward(self, video, train_mode=False, **kwargs):
        return self.forward_train(video, **kwargs) if train_mode else self.forward_test(video, **kwargs)

    def forward_train(self, video, gt_label, **kwargs):
        ret = OrderedDict()

        neck_features = self.neck(self.backbone(video))

        probs = self.head(neck_features)
        loss = self.loss(probs, gt_label)
        ret["loss"] = loss
        with torch.no_grad():
            acc_topk = accuracy(probs, gt_label, self.topk)
        for acc, k in zip(acc_topk, self.topk):
            ret[f"acc@{k}"] = acc
        ret["batch_size"] = video.size(0)
        return ret

    def forward_test(self, video, gt_label=None, **kwargs):
        neck_features = self.neck(self.backbone(video))
        probs = nn.functional.softmax(self.head(neck_features), dim=1)
        if gt_label is None:
            return probs
        ret = OrderedDict()
        acc_topk = accuracy(probs, gt_label, self.topk)
        for acc, k in zip(acc_topk, self.topk):
            ret[f"acc@{k}"] = acc
        ret["batch_size"] = video.size(0)
        return ret
