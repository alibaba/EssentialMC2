# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp

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
    """

    def __init__(self, interval=1, **kwargs):
        priority = kwargs.pop("priority") if "priority" in kwargs else _DEFAULT_CHECKPOINT_PRIORITY
        super(CheckpointHook, self).__init__(priority=priority)
        self.interval = interval

    def before_solve(self, solver):
        if solver.resume_from is None or not osp.exists(solver.resume_from):
            return
        solver.logger.info(f"Loading checkpoint from {solver.resume_from}")
        with FS.get_fs_client(solver.resume_from) as client:
            local_file = client.get_object_to_local_file(solver.resume_from)
            checkpoint = torch.load(local_file)
        solver.load_checkpoint(checkpoint)

    def after_epoch(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return
        if (solver.epoch + 1) % self.interval == 0:
            solver.logger.info(f'Saving checkpoint after {solver.epoch + solver.num_folds} epochs')
            checkpoint = solver.save_checkpoint()
            save_path = osp.join(solver.work_dir, f"epoch-{solver.epoch + solver.num_folds}.pth")

            with FS.get_fs_client(save_path) as client:
                local_file = client.convert_to_local_path(save_path)
                with open(local_file, "wb") as f:
                    torch.save(checkpoint, f)
                client.put_object_from_local_file(local_file, save_path)
