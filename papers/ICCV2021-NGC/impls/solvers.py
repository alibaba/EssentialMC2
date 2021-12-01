# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from typing import Optional

import faiss
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import torch
import torchnet
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

from essmc2.solvers.base_solver import BaseSolver
from essmc2.solvers.registry import SOLVERS
from essmc2.utils.data import transfer_data_to_cuda
from essmc2.utils.model import load_pretrained_dict
from .CalcLargestConnectedComponentsIds import calc_lcc


@SOLVERS.register_class()
class NGCSolver(BaseSolver):
    def __init__(self,
                 model,
                 hyper_params,
                 **kwargs):
        super(NGCSolver, self).__init__(model, **kwargs)
        hyper_params = hyper_params or {}
        self.temperature = hyper_params.get("temperature")
        self.warmup_epoch = hyper_params.get("warmup_epoch")
        self.knn_neighbors = hyper_params.get("knn_neighbors")
        self.low_threshold = hyper_params.get("low_threshold")
        self.high_threshold = hyper_params.get("high_threshold")
        self.do_aug = hyper_params.get("do_aug")
        self.num_classes = hyper_params.get("num_classes")
        self.feature_dim = hyper_params.get("feature_dim")
        self.dataset_name = hyper_params.get("dataset_name")
        self.openset = hyper_params.get("openset")

        # All tensors are on CPU!
        self.hard_labels: Optional[torch.Tensor] = None
        self.soft_labels: Optional[torch.Tensor] = None
        self.clean_ids: Optional[torch.Tensor] = None
        self.temporal_logits: Optional[torch.Tensor] = None
        self.current_logits: Optional[torch.Tensor] = None
        self.prototypes: Optional[torch.Tensor] = None

    def train_step(self, data, warmup=False):
        self.before_iter()

        if self._epoch >= self.warmup_epoch and not warmup:
            # original data: img, img_aug, index, gt_label, meta
            index_bt = data.pop("index")
            gt_label = self.hard_labels[index_bt]  # replace gt_label with hard_labels
            clean_flag = self.clean_ids[index_bt]  # add clean_flag tensor
            data["gt_label"] = gt_label
            data["clean_flag"] = clean_flag
            data_cuda = transfer_data_to_cuda(data)
            result = self.model(**data_cuda, do_aug=self.do_aug, pseudo=True,
                                temperature=self.temperature)
        else:
            data_cuda = transfer_data_to_cuda(data)
            result = self.model(**data_cuda, do_aug=self.do_aug, pseudo=False)

        self._iter_outputs[self._mode] = self._reduce_scalar(result)

        self.after_iter()

    def before_solve(self, data_loaders):
        super().before_solve()

        # init temporal_logits
        N = len(data_loaders['eval'].dataset)
        self.temporal_logits = torch.zeros(N, self.num_classes, dtype=torch.float32)

    def solve(self, data_loaders):
        self.logger.info("Begin to solve...")
        self.before_solve(data_loaders)
        while self._epoch < self.max_epochs:
            self.logger.info(f"Begin to solve at epoch {self._epoch}")
            self.before_epoch()
            self.run_epoch(data_loaders)
            self.after_epoch()
        self.after_solve()
        self.logger.info("Solved.")

    def run_epoch(self, data_loaders):
        # Extract eval dataset features
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], begin to extract eval features...")
        features, labels, probs, logits = self.extract_features(data_loaders["eval"])

        # Build faiss index
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], begin to build searcher and search self...")
        searcher, search_scores, search_ids = self.search_self(features)

        # Do temporal ensemble
        ensemble_alpha = 0.6
        self.temporal_logits = ensemble_alpha * self.temporal_logits + (1. - ensemble_alpha) * logits
        self.current_logits = self.temporal_logits * (1.0 / (1.0 - ensemble_alpha ** (self._epoch + 1)))
        probs = self.current_logits.softmax(dim=1)

        # Clean labels
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], "
                         f"begin to perform noise correction and subgraph selection...")
        clean_labels = torch.from_numpy(np.asarray(data_loaders['eval'].dataset.clean_labels, dtype=np.long)) \
            if self.dataset_name.startswith("cifar") else None
        self.label_clean(search_scores, search_ids,
                         features, labels, probs,
                         clean_labels
                         )

        # Train
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], begin to train model...")
        self.train_mode()
        self.logger.info(f"Begin Train at {self._epoch}...")
        self._epoch_max_iter[self._mode] = len(data_loaders['train'])
        self.before_all_iter()
        for data in data_loaders['train']:
            self.train_step(data)
        self.after_all_iter()

        # Collect prototypes
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], begin to align prototypes and calc auroc...")
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float32)
        if self._epoch >= self.warmup_epoch:
            features_clean = features[self.clean_ids]
            pseudo_labels = self.hard_labels[self.clean_ids]
            for c in range(self.num_classes):
                self.prototypes[c, :] = features_clean[torch.where(pseudo_labels == c)].mean(dim=0)
        else:
            for c in range(self.num_classes):
                self.prototypes[c, :] = features[torch.where(labels == c)].mean(dim=0)
        self.prototypes = torch.nn.functional.normalize(self.prototypes, p=2, dim=1)

        # Test
        self.logger.info(f"Epoch [{self._epoch}/{self.max_epochs}], begin to test model on test dataset...")
        self.test_mode()
        self.logger.info(f"Begin Test at {self._epoch}...")
        self.test(data_loaders['test'], None if 'imagenet' not in data_loaders else data_loaders['imagenet'])

    def search_self(self, features: torch.Tensor):
        res = faiss.StandardGpuResources()
        searcher = faiss.IndexFlatIP(features.size(1))
        gpu_searcher = faiss.index_cpu_to_gpu(res, 0, searcher)
        features_np = features.numpy()
        gpu_searcher.add(features_np)
        search_scores, search_ids = gpu_searcher.search(features_np, self.knn_neighbors + 1)  # Ignore self

        return searcher, search_scores, search_ids

    def extract_features(self, eval_dataloader):
        eval_len = len(eval_dataloader.dataset)
        features = torch.zeros(eval_len, self.feature_dim, dtype=torch.float32)
        labels = torch.zeros(eval_len, dtype=torch.long)
        probs = torch.zeros(eval_len, self.num_classes, dtype=torch.float32)
        logits = torch.zeros(eval_len, self.num_classes, dtype=torch.float32)

        self.eval_mode()
        start_idx = 0
        for data in eval_dataloader:
            with torch.no_grad():
                data_cuda = transfer_data_to_cuda(data)
                o_logits, o_features = self.model(**data_cuda, do_classify=True, do_extract_feature=True)
                o_probs = softmax(o_logits, dim=1)
                bt = o_features.size(0)
                features[start_idx: start_idx + bt] = o_features.cpu()
                labels[start_idx: start_idx + bt] = data["gt_label"]
                probs[start_idx: start_idx + bt] = o_probs.cpu()
                logits[start_idx: start_idx + bt] = o_logits.cpu()
                start_idx += bt

        return features, labels, probs, logits

    def label_clean(self,
                    search_scores: np.ndarray,
                    search_ids: np.ndarray,
                    features: torch.Tensor,
                    labels: torch.Tensor,
                    probs: torch.Tensor,
                    clean_label: Optional[torch.Tensor] = None):
        self.logger.info("Graph-based Noise Correction...")
        sample_nums = features.size(0)
        neighbors = torch.from_numpy(search_ids)

        if self.hard_labels is None:
            pseudo_labels = torch.zeros_like(probs).scatter_(1, labels.view(-1, 1), 1)
        else:
            pseudo_labels = probs.clone()
            labels_clean_one_hot = torch.zeros_like(probs).scatter_(1, self.hard_labels.view(-1, 1), 1)
            pseudo_labels[self.clean_ids] = labels_clean_one_hot[self.clean_ids]
        # norm pseudo labels
        pseudo_labels = pseudo_labels / pseudo_labels.sum(dim=0)
        D = search_scores[:, 1:] ** 3
        I = search_ids[:, 1:]
        row_idx = np.arange(sample_nums)
        # (sample_nums, knn_neighbors),
        row_idx_repeat = np.tile(row_idx, (self.knn_neighbors, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_repeat.flatten('F'), I.flatten('F'))),
                                    shape=(sample_nums, sample_nums))
        W = W + W.T
        # normalize graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1.0 / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D
        # init y vector for each class (eg 5 from the paper) and apply label propagation
        Z = np.zeros((sample_nums, self.num_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - 0.5 * Wn
        for i in range(self.num_classes):
            y = pseudo_labels[:, i]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
            Z[:, i] = f
        # handle numeric errors
        Z[Z < 0] = 0
        self.soft_labels = torch.tensor(Z).float()
        self.soft_labels = self.soft_labels / self.soft_labels.sum(dim=1).reshape(-1, 1)

        # consider the gt label as clean if soft label outputs a score higher than threshold
        self.logger.info("Confidence-based Selection...")
        gt_score = self.soft_labels[labels >= 0, labels]
        gt_clean = gt_score > self.low_threshold
        self.soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), self.num_classes) \
            .scatter_(1, labels[gt_clean].view(-1, 1), 1)

        # get the hard pseudo label and the clean subset used to calculate supervised loss
        max_scores, self.hard_labels = torch.max(self.soft_labels, dim=1)
        self.clean_ids = max_scores > self.high_threshold
        self.logger.info(f'Number of clean samples (confidence-based selection): {self.clean_ids.sum()}')
        self.logger.info("Geometry-based Selection...")
        idx_of_comp_idx2 = calc_lcc(sample_nums, self.num_classes, self.hard_labels.numpy(),
                                    neighbors.numpy(), self.clean_ids.numpy())

        clean_ids_backup = self.clean_ids.clone()
        self.clean_ids = torch.zeros_like(self.clean_ids)
        self.clean_ids[idx_of_comp_idx2] = True
        self.logger.info(f'Number of clean samples (geometry-based selection): {self.clean_ids.sum()}')

        if clean_label is not None:
            acc_meter = torchnet.meter.ClassErrorMeter(topk=[1], accuracy=True)
            acc_meter.add(self.soft_labels, clean_label)
            acc_top1 = acc_meter.value(1)
            self.logger.info(f"Noise correction: accuracy={acc_top1}")

            effective_noise = 1 - 1.0 * (self.hard_labels[clean_ids_backup] == clean_label[
                clean_ids_backup]).sum() / clean_ids_backup.sum()
            self._epoch_outputs["label_clean"]["effective_noise"] = effective_noise.item()
            self.logger.info(f"Confidence-based Selection: effective_noise={effective_noise.item()}")
            effective_noise_lcc = 1 - 1.0 * (
                    self.hard_labels[self.clean_ids] == clean_label[self.clean_ids]).sum() / self.clean_ids.sum()
            self.epoch_outputs["label_clean"]["effective_noise_lcc"] = effective_noise_lcc.item()
            self.logger.info(f"Geometry-based Selection: effective_noise_lcc={effective_noise_lcc.item()}")
            if self.openset:
                ood_noise = torch.sum(clean_label[clean_ids_backup] == -1).item()
                self.epoch_outputs["label_clean"]["ood_noise"] = ood_noise
                self.logger.info(f"Confidence-based Selection: ood_noise={ood_noise}")
                ood_noise_lcc = torch.sum(clean_label[self.clean_ids] == -1).item()
                self.epoch_outputs["label_clean"]["ood_noise_lcc"] = ood_noise_lcc
                self.logger.info(f"Geometry-based Selection: ood_noise_lcc={ood_noise_lcc}")

    def test(self, test_dataloader, imagenet_dataloader=None):
        self.test_mode()
        topk = (1, 5) if self.dataset_name == "webvision" else (1,)
        acc_meter = torchnet.meter.ClassErrorMeter(topk=topk, accuracy=True)

        features = []
        logits = []
        labels = []
        test_auroc = self.dataset_name.startswith("cifar") and self.openset
        with torch.no_grad():
            for data in test_dataloader:
                gt_label_bt = data.pop("gt_label")
                img_bt = data.pop("img")
                data_gpu = transfer_data_to_cuda({"img": img_bt})
                if test_auroc:
                    test_logits, test_features = self.model(**data_gpu, do_extract_feature=True)
                    features.append(test_features.cpu())
                else:
                    test_logits = self.model(**data_gpu, do_extract_feature=False)
                test_logits = test_logits.cpu()
                if test_auroc:
                    logits.append(test_logits)
                    labels.append(gt_label_bt)
                if self.openset:
                    ind_idx = gt_label_bt != -1
                    acc_meter.add(test_logits[ind_idx], gt_label_bt[ind_idx])
                else:
                    acc_meter.add(test_logits, gt_label_bt)
            accuracy_list = acc_meter.value()
            for n, acc in zip(topk, accuracy_list):
                self.logger.info(f"Test: test_accuracy@{n}: {acc}")
                self._epoch_outputs["test"][f"test_accuracy@{n}"] = acc

        if test_auroc and self.epoch >= self.warmup_epoch:
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            with torch.no_grad():
                scores = torch.mm(features.cuda(), self.prototypes.t().cuda()).max(dim=1)[0].cpu()
                try:
                    auc_proto = roc_auc_score((labels != -1).numpy(), scores.numpy())
                    self.logger.info(f"Test: PROTO AUROC {auc_proto}")
                except:
                    self.logger.info(f"Test: Failed to calc proto auroc")

        if imagenet_dataloader is not None:
            acc_meter.reset()
            with torch.no_grad():
                for data in imagenet_dataloader:
                    gt_label_bt = data.pop("gt_label")
                    img_bt = data.pop("img")
                    data_gpu = transfer_data_to_cuda({"img": img_bt})
                    test_logits = self.model(**data_gpu, do_extract_feature=False)
                    test_logits = test_logits.cpu()
                    if self.openset:
                        ind_idx = gt_label_bt != -1
                        acc_meter.add(test_logits[ind_idx], gt_label_bt[ind_idx])
                    else:
                        acc_meter.add(test_logits, gt_label_bt)
                accuracy_list = acc_meter.value()
                for n, acc in zip(topk, accuracy_list):
                    self.logger.info(f"Test imagenet: test_accuracy@{n}: {acc}")
                    self._epoch_outputs["test"][f"imagenet_test_accuracy@{n}"] = acc

    def load_checkpoint(self, checkpoint: dict):
        self._epoch = checkpoint["epoch"]
        for mode_name, iter_value in checkpoint["total_iter"].items():
            self._total_iter[mode_name] = iter_value
        load_pretrained_dict(self.model, checkpoint["state_dict"], self.logger)
        self.optimizer.load_state_dict(checkpoint["checkpoint"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.hard_labels = checkpoint["tensors"]["hard_labels"]
        self.soft_labels = checkpoint["tensors"]["soft_labels"]
        self.clean_ids = checkpoint["tensors"]["clean_ids"]
        self.temporal_logits = checkpoint["tensors"]["temporal_logits"]
        self.current_logits = checkpoint["tensors"]["current_logits"]
        self.prototypes = checkpoint["tensors"]["prototypes"]
        self._epoch += 1  # Move to next epoch

    def save_checkpoint(self) -> dict:
        checkpoint = {
            "epoch": self._epoch,
            "total_iter": self._total_iter,
            "state_dict": self.model.state_dict(),
            "checkpoint": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "tensors": {
                "hard_labels": self.hard_labels,
                "soft_labels": self.soft_labels,
                "clean_ids": self.clean_ids,
                "temporal_logits": self.temporal_logits,
                "current_logits": self.current_logits,
                "prototypes": self.prototypes
            }}
        return checkpoint
