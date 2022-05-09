# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .hook import Hook
from .registry import HOOKS

_DEFAULT_LR_PRIORITY = 200


def _get_lr_from_scheduler(lr_scheduler, cur_epoch):
    """Ugly solution to get lr by epoch.
    PyTorch lr scheduler get_lr() function is recommended to call in step()
    Here we mock the environment.

    Args:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler):
        cur_epoch (number): int or float (when num_folds > 1)

    Returns:
        Learning rate at cur_epoch.
    """
    lr_scheduler._get_lr_called_within_step = True
    last_epoch_bk = lr_scheduler.last_epoch
    lr_scheduler.last_epoch = cur_epoch
    if hasattr(lr_scheduler, "_get_closed_form_lr"):
        lr = lr_scheduler._get_closed_form_lr()[0]
    else:
        lr = lr_scheduler.get_lr()[0]
    lr_scheduler._get_lr_called_within_step = False
    lr_scheduler.last_epoch = last_epoch_bk
    return lr


@HOOKS.register_class()
class LrHook(Hook):
    """ Learning rate updater hook.
    If warmup, warmup_end_lr will be calculated by lr_scheduler at warmup_epochs.
    Lr in warmup period is set based on warmup_func.
    If set_by_epoch, lr is set at end of epoch. Otherwise, lr is set before training iteration.

    Args:
        set_by_epoch (bool): Reset learning rate by epoch, we recommend true if solver.num_folds == 1
        warmup_func (str, None): Do not warm up if None, currently support linear warmup
        warmup_epochs (int):
        warmup_start_lr (float):
    """

    def __init__(self,
                 set_by_epoch=True,
                 warmup_func=None,
                 warmup_epochs=1,
                 warmup_start_lr=0.0001,
                 **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_LR_PRIORITY
        super(LrHook, self).__init__(priority=priority)
        self.warmup_func = warmup_func
        if self.warmup_func is not None:
            assert self.warmup_func in ("linear",), "Only linear warmup supported"
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = 0
        self.set_by_epoch = set_by_epoch

    def before_solve(self, solver):
        # Not support warmup for multiple lr schedulers
        if self.warmup_func is not None and not isinstance(solver.lr_scheduler, dict):
            self.warmup_end_lr = _get_lr_from_scheduler(solver.lr_scheduler, self.warmup_epochs)
            for param_group in solver.optimizer.param_groups:
                param_group["lr"] = self.warmup_start_lr

    def _get_warmup_lr(self, cur_epoch):
        # if self.warmup_func == "linear":
        alpha = (self.warmup_end_lr - self.warmup_start_lr) / self.warmup_epochs
        return self.warmup_start_lr + alpha * cur_epoch

    def after_epoch(self, solver):
        if solver.lr_scheduler is not None and self.set_by_epoch:
            if not isinstance(solver.lr_scheduler, dict):
                last_lr = solver.optimizer.param_groups[0]["lr"]
                for _ in range(solver.num_folds):
                    solver.lr_scheduler.step()
                new_lr = solver.optimizer.param_groups[0]["lr"]
                if self.warmup_func is not None and solver.epoch < self.warmup_epochs:
                    new_lr = self._get_warmup_lr(solver.epoch)
                    for param_group in solver.optimizer.param_groups:
                        param_group["lr"] = new_lr
                if last_lr != new_lr:
                    solver.logger.info(f"Change learning rate from {last_lr} to {new_lr}")
                else:
                    solver.logger.info(f"Keep learning rate = {last_lr}")
            else:
                for key, sub_lr_scheduler in solver.lr_scheduler.items():
                    last_lr = solver.optimizer[key].param_groups[0]['lr']
                    for _ in range(solver.num_folds):
                        sub_lr_scheduler.step()
                    new_lr = solver.optimizer[key].param_groups[0]['lr']
                    if last_lr != new_lr:
                        solver.logger.info(f"Change {key} learning rate from {last_lr} to {new_lr}")
                    else:
                        solver.logger.info(f"Keep {key} learning rate = {last_lr}")

    def before_iter(self, solver):
        if solver.lr_scheduler is not None and not self.set_by_epoch and solver.is_train_mode:
            cur_epoch_float = solver.epoch + solver.num_folds * solver.iter / solver.epoch_max_iter
            if not isinstance(solver.lr_scheduler, dict):
                if self.warmup_func is not None and cur_epoch_float < self.warmup_epochs:
                    new_lr = self._get_warmup_lr(cur_epoch_float)
                else:
                    new_lr = _get_lr_from_scheduler(solver.lr_scheduler, cur_epoch_float)
                for param_group in solver.optimizer.param_groups:
                    param_group["lr"] = new_lr
            else:
                for key, sub_lr_scheduler in solver.lr_scheduler.items():
                    new_lr = _get_lr_from_scheduler(sub_lr_scheduler, cur_epoch_float)
                    for param_group in solver.optimizer[key].param_groups:
                        param_group["lr"] = new_lr
