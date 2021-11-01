# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

from essmc2.models import TrainModule, MODELS, BACKBONES, NECKS, HEADS
from essmc2.utils.metric import accuracy


@MODELS.register_class()
class MoSINet(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 label_mode='joint',
                 freeze_bn=False,
                 topk=(1,)):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.label_mode = label_mode
        if self.label_mode == "joint":
            self.head = HEADS.build(head)
        else:
            self.head_x = HEADS.build(head)
            self.head_y = HEADS.build(head)
        self.freeze_bn = freeze_bn

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = topk

        self.loss = torch.nn.CrossEntropyLoss()

    def train(self, mode=True):
        self.training = mode
        super(MoSINet, self).train(mode=mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.freeze_bn:
                module.train(False)
        return self

    def forward(self, video, train_mode=False, **kwargs):
        return self.forward_train(video, **kwargs) if train_mode else self.forward_test(video, **kwargs)

    def forward_train(self, video, mosi_label, **kwargs):
        ret = OrderedDict()

        b, n, c, t, h, w = video.shape
        video = video.reshape(b * n, c, t, h, w)

        neck_features = self.neck(self.backbone(video))

        if self.label_mode == "joint":
            probs = self.head(neck_features)
            labels = mosi_label["move_joint"].reshape(probs.shape[0])
            loss = self.loss(probs, labels)
            ret["loss"] = loss
            with torch.no_grad():
                acc_topk = accuracy(probs, labels, self.topk)
            for acc, k in zip(acc_topk, self.topk):
                ret[f"acc@{k}"] = acc
        else:
            probs_x = self.head_x(neck_features)
            probs_y = self.head_y(neck_features)
            labels_x = mosi_label["move_x"].reshape(probs_x.shape[0])
            labels_y = mosi_label["move_y"].reshape(probs_y.shape[0])
            loss_x = self.loss(probs_x, labels_x)
            loss_y = self.loss(probs_y, labels_y)
            loss = loss_x + loss_y
            ret["loss"] = loss
            ret["loss_x"] = loss_x
            ret["loss_y"] = loss_y
            with torch.no_grad():
                acc_topk_x = accuracy(probs_x, labels_x, self.topk)
            for acc, k in zip(acc_topk_x, self.topk):
                ret[f"acc_x@{k}"] = acc
            with torch.no_grad():
                acc_topk_y = accuracy(probs_y, labels_y, self.topk)
            for acc, k in zip(acc_topk_y, self.topk):
                ret[f"acc_y@{k}"] = acc

        ret["batch_size"] = video.size(0)

        return ret

    def forward_test(self, video, mosi_label=None, **kwargs):
        b, n, c, t, h, w = video.shape
        video = video.reshape(b * n, c, t, h, w)

        neck_features = self.neck(self.backbone(video))
        if self.label_mode == "joint":
            probs = nn.functional.softmax(self.head(neck_features), dim=1)
            if mosi_label is None:
                return probs
            ret = OrderedDict()
            labels = mosi_label["move_joint"].reshape(probs.shape[0])
            acc_topk = accuracy(probs, labels, self.topk)
            for acc, k in zip(acc_topk, self.topk):
                ret[f"acc@{k}"] = acc
            return ret
        else:
            probs_x = nn.functional.softmax(self.head_x(neck_features), dim=1)
            probs_y = nn.functional.softmax(self.head_y(neck_features), dim=1)
            if mosi_label is None:
                return probs_x, probs_y
            ret = OrderedDict()
            labels_x = mosi_label["move_x"].reshape(probs_x.shape[0])
            labels_y = mosi_label["move_y"].reshape(probs_y.shape[0])
            acc_topk_x = accuracy(probs_x, labels_x, self.topk)
            acc_topk_y = accuracy(probs_y, labels_y, self.topk)

            for acc, k in zip(acc_topk_x, self.topk):
                ret[f"acc_x@{k}"] = acc
            for acc, k in zip(acc_topk_y, self.topk):
                ret[f"acc_y@{k}"] = acc

            ret["batch_size"] = video.size(0)
            return ret
