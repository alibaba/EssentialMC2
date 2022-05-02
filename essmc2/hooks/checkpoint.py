# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import os.path as osp
import sys
import warnings

import torch
import torch.distributed as du

from essmc2.utils.file_systems import FS
from .hook import Hook
from .registry import HOOKS

_DEFAULT_CHECKPOINT_PRIORITY = 300


@HOOKS.register_class()
class CheckpointHook(Hook):
    """ Checkpoint resume or save hook.

    Args:
        interval (int): Save interval, by epoch.
        save_best (bool): Save the best checkpoint by a metric key, default is False.
        save_best_by (str): How to get the best the checkpoint by the metric key, default is ''.
            + means the higher the best (default).
            - means the lower the best.
            E.g. +acc@1, -err@1, acc@5(same as +acc@5)
    """

    def __init__(self,
                 interval=1,
                 save_best=False,
                 save_best_by="",
                 **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_CHECKPOINT_PRIORITY
        super(CheckpointHook, self).__init__(priority=priority)
        self.interval = interval
        self.save_best = save_best
        self.save_best_by = save_best_by
        if self.save_best and not self.save_best_by:
            warnings.warn("CheckpointHook: Parameter 'save_best_by' is not set, turn off save_best function.")
            self.save_best = False
        self.higher_the_best = True
        if self.save_best:
            if self.save_best_by.startswith("+"):
                self.save_best_by = self.save_best_by[1:]
            elif self.save_best_by.startswith("-"):
                self.save_best_by = self.save_best_by[1:]
                self.higher_the_best = False
        if self.save_best and not self.save_best_by:
            warnings.warn("CheckpointHook: Parameter 'save_best_by' is not valid, turn off save_best function.")
            self.save_best = False
        self._last_best = None if not self.save_best else (
            sys.float_info.min if self.higher_the_best else sys.float_info.max
        )

    def before_solve(self, solver):
        if solver.resume_from is None:
            return
        if not FS.exists(solver.resume_from):
            solver.logger.error(f"File not exists {solver.resume_from}")
            return

        with FS.get_from(solver.resume_from) as local_file:
            solver.logger.info(f"Loading checkpoint from {solver.resume_from}")
            checkpoint = torch.load(local_file)

        solver.load_checkpoint(checkpoint)
        if self.save_best and "_CheckpointHook_best" in checkpoint:
            self._last_best = checkpoint["_CheckpointHook_best"]

    def after_epoch(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return
        if (solver.epoch + solver.num_folds) % self.interval == 0:
            solver.logger.info(f'Saving checkpoint after {solver.epoch + solver.num_folds} epochs')
            checkpoint = solver.save_checkpoint()
            if checkpoint is None or len(checkpoint) == 0:
                return
            cur_is_best = False
            if self.save_best:
                # Try to get current state from epoch_outputs["eval"]
                cur_state = None \
                    if self.save_best_by not in solver.epoch_outputs["eval"] \
                    else solver.epoch_outputs["eval"][self.save_best_by]
                # Try to get current state from agg_iter_outputs["eval"] if do_final_eval is False
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs["eval"] \
                        else solver.agg_iter_outputs["eval"][self.save_best_by]
                # Try to get current state from agg_iter_outputs["train"] if no evaluation
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs["train"] \
                        else solver.agg_iter_outputs["train"][self.save_best_by]
                if cur_state is not None:
                    if self.higher_the_best and cur_state > self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    elif not self.higher_the_best and cur_state < self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    checkpoint["_CheckpointHook_best"] = self._last_best
            # minus 1, means index
            save_path = osp.join(solver.work_dir, "epoch-{:05d}.pth".format(solver.epoch + solver.num_folds))

            with FS.put_to(save_path) as local_file:
                with open(local_file, "wb") as f:
                    torch.save(checkpoint, f)
            if cur_is_best:
                best_path = osp.join(solver.work_dir, f"best.pth")
                with FS.get_fs_client(best_path) as client:
                    client.make_link(best_path, save_path)
