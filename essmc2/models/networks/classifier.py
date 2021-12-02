# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict
from functools import partial

import torch.nn as nn
from torch.nn.functional import softmax, sigmoid

from essmc2.models.networks.train_module import TrainModule
from essmc2.models.registry import MODELS, BACKBONES, NECKS, HEADS, LOSSES
from essmc2.utils.metrics import METRICS

_ACTIVATE_MAPPER = {
    "softmax": partial(softmax, dim=1),
    "sigmoid": sigmoid
}


@MODELS.register_class()
class Classifier(TrainModule):
    """ Base classifier implementation.

    Args:
        backbone (dict): Defines backbone.
        neck (dict, optional): Defines neck. Use Identity if none.
        head (dict): Defines head.
        act_name (str): Defines activate function, 'softmax' or 'sigmoid'.
        topk (Sequence[int]): Defines how to calculate accuracy metrics.
        freeze_bn (bool): If True, freeze all BatchNorm layers including LayerNorm.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 loss=None,
                 act_name='softmax',
                 topk=(1,),
                 freeze_bn=False):
        super().__init__()
        # Construct model
        self.backbone: nn.Module = BACKBONES.build(backbone)
        self.neck: nn.Module = NECKS.build(neck or dict(type="Identity"))
        assert head is not None
        self.head: nn.Module = HEADS.build(head)

        # Construct loss
        self.loss = LOSSES.build(loss or dict(type='CrossEntropy'))

        # Construct activate function
        self.act_fn = _ACTIVATE_MAPPER[act_name]
        self.metric = METRICS.build(dict(type='accuracy', topk=topk))

        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        self.training = mode
        super(Classifier, self).train(mode=mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    module.train(False)
        return self

    def forward(self, img, gt_label=None, **kwargs):
        return self.forward_train(img, gt_label=gt_label) \
            if self.training else self.forward_test(img, gt_label=gt_label)

    def forward_train(self, img, gt_label=None):
        probs = self.head(self.neck(self.backbone(img)))
        if gt_label is None:
            return probs

        ret = OrderedDict()
        loss = self.loss(probs, gt_label)
        ret["loss"] = loss
        ret["batch_size"] = img.size(0)
        ret.update(self.metric(probs, gt_label))
        return ret

    def forward_test(self, img, gt_label=None):
        logits = self.act_fn(self.head(self.neck(self.backbone(img))))
        if gt_label is not None:
            ret = OrderedDict()
            ret["logits"] = logits
            ret["batch_size"] = img.size(0)
            ret.update(self.metric(logits, gt_label))
            return ret
        return logits


@MODELS.register_class()
class VideoClassifier(Classifier):
    """ Classifier for video.
    Default input tensor is video.

    """

    def forward(self, video, gt_label=None, **kwargs):
        return self.forward_train(video, gt_label=gt_label) \
            if self.training else self.forward_test(video, gt_label=gt_label)


@MODELS.register_class()
class VideoClassifier2x(VideoClassifier):
    """ A 2-way classifier for video.

    """

    def forward_train(self, video, gt_label=None):
        probs0, probs1 = self.head(self.neck(self.backbone(video)))
        if gt_label is not None:
            ret = OrderedDict()
            loss = self.loss(probs0, gt_label[:, 0]) + self.loss(probs1, gt_label[:, 1])
            ret["loss"] = loss
            ret["batch_size"] = video.size(0)
            acc_0 = self.metric(probs0, gt_label[:, 0])
            acc_0 = {key.relace("@", "_0@"): value for key, value in acc_0.items()}
            acc_1 = self.metric(probs1, gt_label[:, 1])
            acc_1 = {key.relace("@", "_1@"): value for key, value in acc_1.items()}
            ret.update(acc_0)
            ret.update(acc_1)
            return ret
        return {
            "logits0": self.act_fn(probs0),
            "logits1": self.act_fn(probs1)
        }

    def forward_test(self, video, gt_label=None):
        probs0, probs1 = self.head(self.neck(self.backbone(video)))
        logits0, logits1 = self.act_fn(probs0), self.act_fn(probs1)
        if gt_label is None:
            return {
                "logits0": logits0,
                "logits1": logits1
            }
        ret = OrderedDict()
        ret["logits0"] = logits0
        ret["logits1"] = logits1
        acc_0 = self.metric(probs0, gt_label[:, 0])
        acc_0 = {key.relace("@", "_0@"): value for key, value in acc_0.items()}
        acc_1 = self.metric(probs1, gt_label[:, 1])
        acc_1 = {key.relace("@", "_1@"): value for key, value in acc_1.items()}
        ret.update(acc_0)
        ret.update(acc_1)
        return ret
