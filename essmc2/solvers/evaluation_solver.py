# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import torch

from .base_solver import BaseSolver
from .registry import SOLVERS
from ..utils.data import transfer_data_to_cuda


@SOLVERS.register_class()
class EvaluationSolver(BaseSolver):
    """ Evaluation solver once.

    Args:
        model (torch.nn.Module): Model to train or eval.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

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
        self.max_epochs = 1
        assert "val" in data_loaders
        self.logger.info(f"Begin to evaluate...")
        self.run_eval_epoch(data_loaders["val"])

    def load_checkpoint(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["state_dict"])

    def save_checkpoint(self) -> dict:
        raise NotImplementedError
