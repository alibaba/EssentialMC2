# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from essmc2.models import TrainModule
from essmc2.models.registry import MODELS, BACKBONES, NECKS, HEADS
from essmc2.utils.metric import accuracy


class NGCInference(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 head):
        super(NGCInference, self).__init__()
        self.backbone = BACKBONES.build(backbone)
        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)

    def forward(self, img, do_classify=True, do_extract_feature=False):
        return self.head(self.neck(self.backbone(img)), do_classify=do_classify, do_extract_feature=do_extract_feature)


@MODELS.register_class()
class NGCNetwork(TrainModule):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 num_classes,
                 alpha=8.0,
                 w_inst=1.0,
                 w_sup=1.0,
                 data_parallel=False,
                 **kwargs):
        super(NGCNetwork, self).__init__()

        self.infer = NGCInference(backbone, neck, head)
        self.loss_instance = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.w_inst = w_inst
        self.w_sup = w_sup
        self.num_classes = num_classes

        if data_parallel:
            self.infer = nn.parallel.DataParallel(self.infer)

    def forward(self, img, **kwargs):
        return self.forward_train(img, **kwargs) if self.training else self.forward_test(img, **kwargs)

    def forward_train(self, img, gt_label=None, img_aug=None,
                      clean_flag=None, do_aug=False, pseudo=False,
                      temperature=0.3,
                      **kwargs):
        """
        :param img: Tensor, (N, 3, H, W)
        :param gt_label: Tensor(optional), (N)
        :param img_aug: Tensor(optional), (N, 3, H, W)
        :param clean_flag: Tensor(optional), (N)
        :param do_aug: bool
        :param pseudo: bool
        :param temperature: float
        :param kwargs:
        :return: dict, which contains at least `loss`
        """
        assert gt_label is not None
        assert img_aug is not None
        result = OrderedDict()
        batch_size = img.size(0)

        # Get accuracy NOW!
        if pseudo:
            logits_orig, feature_orig = self.infer(img, do_classify=True, do_extract_feature=True)
        else:
            # Only for calculating accuracy
            with torch.no_grad():
                logits_orig = self.infer(img, do_classify=True, do_extract_feature=False)

        with torch.no_grad():
            acc = accuracy(logits_orig, gt_label, topk=(1,))[0]
            result['acc_cls'] = acc.item()

        # Get one hot labels
        one_hot_labels = torch.zeros(batch_size, self.num_classes, device=gt_label.device) \
            .scatter_(1, gt_label.view(-1, 1), 1)  # (N, C)

        # Get clean data if needed
        if clean_flag is not None:
            img_clean = img[clean_flag]
            img_aug_clean = img_aug[clean_flag]
            one_hot_labels_clean = one_hot_labels[clean_flag]
        else:
            img_clean = img
            img_aug_clean = img_aug
            one_hot_labels_clean = one_hot_labels

        # If use augmentation inputs, concat them with original inputs
        if do_aug:
            inputs = torch.cat([img_clean, img_aug_clean], dim=0)
            targets = torch.cat([one_hot_labels_clean, one_hot_labels_clean], dim=0)
        else:
            inputs = img_clean
            targets = one_hot_labels_clean

        # Do mix up
        idx = torch.randperm(inputs.size(0))
        param_l = np.random.beta(self.alpha, self.alpha)
        input_mix = param_l * inputs + (1.0 - param_l) * inputs[idx]
        targets_mix = param_l * targets + (1.0 - param_l) * targets[idx]
        output_mix = self.infer(input_mix, do_classify=True, do_extract_feature=False)

        # Get classification loss
        loss = -torch.mean(torch.sum(F.log_softmax(output_mix, dim=1) * targets_mix, dim=1))

        result["loss_cls"] = loss.item()

        if pseudo:
            # Do Instance Contrastive Loss
            # Shuf the image augmentation instances to avoid learning the sequence position
            shuf_idx = torch.randperm(batch_size)
            mapping = {k: v for v, k in enumerate(shuf_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])
            img_aug_shuf = img_aug[shuf_idx]
            feature_aug = self.infer(img_aug_shuf, do_classify=False, do_extract_feature=True)
            feature_aug = feature_aug[reverse_idx]
            # Calc similarity
            similarity = torch.mm(feature_orig, feature_orig.t())  # (N, N)
            mask = (torch.ones_like(similarity) - torch.eye(batch_size, device=similarity.device)).bool()
            similarity = similarity.masked_select(mask).view(batch_size, -1)  # (N, N-1)
            similarity_aug = torch.mm(feature_orig, feature_aug.t())
            similarity_aug = similarity_aug.masked_select(mask).view(batch_size, -1)  # (N, N-1)
            # logits
            logits_pos = torch.bmm(feature_orig.view(batch_size, 1, -1), feature_aug.view(batch_size, -1, 1)) \
                .squeeze(-1)  # (N, 1)
            logits_neg = torch.cat([similarity, similarity_aug], dim=1)  # (N, 2N-2)
            logits = torch.cat([logits_pos, logits_neg], dim=1)  # (N, 2N-1)
            instance_labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)  # (N, )
            loss_instance = self.loss_instance(logits / temperature, instance_labels)
            acc_instance = accuracy(logits, instance_labels)[0]
            result["loss_inst"] = loss_instance.item()
            result["acc_inst"] = acc_instance.item()

            # Do Supervised Contrastive Loss
            clean_flag_new = clean_flag.view(-1, 1)  # (N, 1)
            clean_mask = torch.eq(clean_flag_new, clean_flag_new.T).float() * clean_flag_new.view(1, -1)  # (N, N)
            tmp_mask = (torch.ones_like(clean_mask) - torch.eye(batch_size, device=clean_mask.device)).bool()
            clean_mask = clean_mask.masked_select(tmp_mask).view(batch_size, -1)  # (N, N-1)
            clean_mask = torch.cat(
                (torch.ones(batch_size, device=clean_mask.device).view(-1, 1), clean_mask, clean_mask),
                dim=1)  # (N, 2N-1), clean flag to logits_icl
            gt_label_new = gt_label.view(-1, 1)  # (N, 1)
            inst_labels = torch.eq(gt_label_new, gt_label_new.T).float()  # (N, N)
            inst_mask = (torch.ones_like(inst_labels) - torch.eye(batch_size, device=inst_labels.device)).bool()
            inst_labels = inst_labels.masked_select(inst_mask).view(batch_size, -1)
            inst_labels = torch.cat(
                (torch.ones(batch_size, device=inst_labels.device).view(-1, 1), inst_labels, inst_labels),
                dim=1)  # (N, 2N-1), labels to logits_icl
            inst_labels = inst_labels * clean_mask  # (N, 2N-1), only use the clean instances
            log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
            mean_log_prob_pos = (inst_labels * log_prob).sum(1) / inst_labels.sum(1)
            loss_sup = -1 * mean_log_prob_pos.view(1, batch_size).mean()

            result["loss_sup"] = loss_sup.item()

            loss = loss + self.w_inst * loss_instance + self.w_sup * loss_sup

        result["loss"] = loss

        return result

    def forward_test(self, img, do_classify=True, do_extract_feature=False, **kwargs):
        return self.infer(img, do_classify=do_classify, do_extract_feature=do_extract_feature)
