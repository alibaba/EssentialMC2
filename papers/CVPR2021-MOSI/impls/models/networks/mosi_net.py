# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import torch.nn as nn

from essmc2.models import TrainModule, MODELS, BACKBONES, NECKS, HEADS, LOSSES
from essmc2.utils.metrics import METRICS


@MODELS.register_class()
class MoSINet(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 loss=None,
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

        self.metric = METRICS.build(dict(type='accuracy', topk=topk))

        self.loss = LOSSES.build(loss or dict(type="CrossEntropy"))

    def train(self, mode=True):
        self.training = mode
        super(MoSINet, self).train(mode=mode)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)) and self.freeze_bn:
                module.train(False)
        return self

    def forward(self, video, mosi_label=None, **kwargs):
        return self.forward_train(video, mosi_label) if self.training else self.forward_test(video, mosi_label)

    def forward_train(self, video, mosi_label=None):
        if len(video.shape) == 6:
            b, n, c, t, h, w = video.shape
            video = video.reshape(b * n, c, t, h, w)

        neck_features = self.neck(self.backbone(video))

        if mosi_label is None:
            if self.label_mode == "joint":
                probs = self.head(neck_features)
                return probs
            else:
                probs_x = self.head_x(neck_features)
                probs_y = self.head_y(neck_features)
                return {
                    "probs_x": probs_x,
                    "probs_y": probs_y
                }

        ret = OrderedDict()
        if self.label_mode == "joint":
            probs = self.head(neck_features)
            labels = mosi_label["move_joint"].reshape(probs.shape[0])
            loss = self.loss(probs, labels)
            ret["loss"] = loss
            ret.update(self.metric(probs, labels))
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
            ret.update(self.metric(probs_x, labels_x, prefix="acc_x"))
            ret.update(self.metric(probs_y, labels_y, prefix="acc_y"))

        ret["batch_size"] = video.size(0)

        return ret

    def forward_test(self, video, mosi_label=None):
        if len(video.shape) == 6:
            b, n, c, t, h, w = video.shape
            video = video.reshape(b * n, c, t, h, w)

        neck_features = self.neck(self.backbone(video))

        if mosi_label is not None:
            ret = OrderedDict()
            if self.label_mode == "joint":
                logits = nn.functional.softmax(self.head(neck_features), dim=1)
                ret["logits"] = logits
                ret.update(self.metric(logits, mosi_label["move_joint"].reshape(logits.shape[0])))
                return ret
            else:
                logits_x = nn.functional.softmax(self.head_x(neck_features), dim=1)
                logits_y = nn.functional.softmax(self.head_y(neck_features), dim=1)
                ret["logits_x"] = logits_x
                ret["logits_y"] = logits_y
                labels_x = mosi_label["move_x"].reshape(logits_x.shape[0])
                labels_y = mosi_label["move_y"].reshape(logits_y.shape[0])
                ret.update(self.metric(logits_x, labels_x, prefix="acc_x"))
                ret.update(self.metric(logits_y, labels_y, prefix="acc_y"))
                ret["batch_size"] = video.size(0)
                return ret

        if self.label_mode == "joint":
            return nn.functional.softmax(self.head(neck_features), dim=1)
        else:
            return {
                "logits_x": nn.functional.softmax(self.head_x(neck_features), dim=1),
                "logits_y": nn.functional.softmax(self.head_y(neck_features), dim=1)
            }
