# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch

from .base_solver import BaseSolver
from .registry import SOLVERS
from ..utils.data import transfer_data_to_cuda


@SOLVERS.register_class()
class TrainValSolver(BaseSolver):
    """ Standard train and eval steps solver

    Args:
        model (torch.nn.Module): Model to train or eval.
        eval_interval (int): Interval between epochs, default is 1.

    """

    def __init__(self, model, eval_interval=1, **kwargs):
        super().__init__(model, **kwargs)
        self.eval_interval = eval_interval

    def run_train_epoch(self, train_data_loader):
        self.train_mode()
        self.before_all_iter()
        self._epoch_max_iter = len(train_data_loader)
        for data in train_data_loader:
            self.before_iter()
            data_gpu = transfer_data_to_cuda(data)
            self._iter_outputs = self.model(**data_gpu)
            self.after_iter()
        self.after_all_iter()

    @torch.no_grad()
    def run_eval_epoch(self, val_data_loader):
        self.eval_mode()
        self._iter = 0
        self._epoch_max_iter = len(val_data_loader)
        self.before_all_iter()
        for data in val_data_loader:
            self.before_iter()
            data_gpu = transfer_data_to_cuda(data)
            self._iter_outputs = self.model(**data_gpu)
            self.after_iter()
        self.after_all_iter()

    def run_epoch(self, data_loaders):
        self.logger.info(f"Begin to train at Epoch [{self._epoch}/{self.max_epochs}]...")
        self.run_train_epoch(data_loaders["train"])

        if "val" in data_loaders and (
                (self._epoch + 1) % self.eval_interval == 0 or self._epoch == self.max_epochs - 1):
            self.logger.info(f"Begin to val at Epoch [{self._epoch}/{self.max_epochs}]...")
            self.run_eval_epoch(data_loaders["val"])

    def load_checkpoint(self, checkpoint: dict):
        self._epoch = checkpoint["epoch"]
        self._total_train_iter = checkpoint["total_train_iter"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["checkpoint"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self._epoch += 1  # Move to next epoch

    def save_checkpoint(self) -> dict:
        checkpoint = {
            "epoch": self._epoch,
            "total_train_iter": self._total_train_iter,
            "state_dict": self.model.state_dict(),
            "checkpoint": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        return checkpoint
