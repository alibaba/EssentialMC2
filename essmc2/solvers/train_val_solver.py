# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch

from .evaluation_solver import EvaluationSolver
from .registry import SOLVERS
from ..utils.data import transfer_data_to_cuda


@SOLVERS.register_class()
class TrainValSolver(EvaluationSolver):
    """ Standard train and eval steps solver

    Args:
        model (torch.nn.Module): Model to train or eval.

    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def run_train_epoch(self, train_data_loader):
        self.train_mode()
        self.before_all_iter()
        self._epoch_max_iter[self._mode] = len(train_data_loader)
        for data in train_data_loader:
            self.before_iter()
            self._iter_inputs[self._mode] = transfer_data_to_cuda(data)
            self._iter_outputs[self._mode] = self._reduce_scalar(self.model(**self._iter_inputs[self._mode]))
            self.after_iter()
        self.after_all_iter()

    def run_epoch(self, data_loaders):
        self.logger.info(f"Begin to train at Epoch [{self._epoch}/{self.max_epochs}]...")
        self.run_train_epoch(data_loaders["train"])

        if "eval" in data_loaders and (
                (self._epoch + self.num_folds) % self.eval_interval == 0 or self._epoch == self.max_epochs - 1):
            self.logger.info(f"Begin to evaluate at Epoch [{self._epoch}/{self.max_epochs}]...")
            self.run_eval_epoch(data_loaders["eval"])

    def load_checkpoint(self, checkpoint: dict):
        self._epoch = checkpoint["epoch"]
        for mode_name, total_iter in checkpoint["total_iters"].items():
            self._total_iter[mode_name] = total_iter
        self.model.load_state_dict(checkpoint["state_dict"])

        if isinstance(self.optimizer, dict):
            for key, value in checkpoint['optimizer'].items():
                self.optimizer[key].load_state_dict(value)
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                for key, value in checkpoint['lr_scheduler'].items():
                    self.lr_scheduler[key].load_state_dict(value)
            else:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self._epoch += self.num_folds  # Move to next epoch

    def save_checkpoint(self) -> dict:
        checkpoint = {
            "epoch": self._epoch,
            "total_iters": self._total_iter,
            "state_dict": self.model.state_dict(),
        }

        if isinstance(self.optimizer, dict):
            optimizer_state_dict = {}
            for key, value in self.optimizer.items():
                optimizer_state_dict[key] = value.state_dict()
            checkpoint['optimizer'] = optimizer_state_dict
        else:
            checkpoint['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                lr_scheduler_state_dict = {}
                for key, value in self.lr_scheduler.items():
                    lr_scheduler_state_dict[key] = value.state_dict()
                checkpoint['lr_scheduler'] = lr_scheduler_state_dict
            else:
                checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        return checkpoint
