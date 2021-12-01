# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import os.path as osp
from collections import defaultdict, OrderedDict

import torch

from .base_solver import BaseSolver
from .registry import SOLVERS
from ..utils.data import transfer_data_to_cuda, transfer_data_to_cpu
from ..utils.distribute import gather_data
from ..utils.distribute import get_dist_info
from ..utils.file_systems import FS
from ..utils.metrics import METRICS


@SOLVERS.register_class()
class EvaluationSolver(BaseSolver):
    """ Evaluation once.

    1. Inference results and ground truth data such as gt_label will be collected,
        according to metric_cfg.keys
    2. Finally, metric functions will be invoked on related keys.

    Notice all result tensors can be placed on ONE GPU device.

    Args:
        model (torch.nn.Module): Model to train or eval.
        do_final_eval (bool): If True, collect all results according to metric_cfg
            and calculate metric values in the end. Default is False.
        meta_keys (tuple or list, optional): Collect keys from input metas.
        eval_metric_cfg (dict, Sequence, optional): Metric function descriptor like
            {
                "metric": dict(type="accuracy", topk=(1, )),
                "keys": ("result", "gt_label")
            }
        save_eval_data (bool): If True, save all collected data. Default path is
            "WORK_DIR/eval_{epoch}_data.pth"

    """

    def __init__(self,
                 model,
                 eval_interval=1,
                 do_final_eval=False,
                 meta_keys=None,
                 eval_metric_cfg=None,
                 save_eval_data=False,
                 **kwargs):
        super().__init__(model, **kwargs)

        self.eval_interval = eval_interval
        self.metrics = []
        self.meta_keys = set(meta_keys or [])
        self.metric_keys = set()
        self._build_metrics(eval_metric_cfg)
        self.meta_keys = sorted(list(self.meta_keys))
        self.metric_keys = sorted(list(self.metric_keys))
        self.do_final_eval = do_final_eval
        self.save_eval_data = save_eval_data

    @torch.no_grad()
    def run_eval_epoch(self, val_data_loader):
        if (self.epoch + 1) % self.eval_interval != 0:
            return

        collect_data = defaultdict(list)

        rank, world_size = get_dist_info()

        # Enter evaluate mode
        self.eval_mode()
        self._epoch_max_iter[self._mode] = len(val_data_loader)
        self.before_all_iter()
        for data in val_data_loader:
            self.before_iter()
            data_gpu = transfer_data_to_cuda(data)
            result = self.model(**data_gpu)

            self._iter_outputs[self._mode] = self._reduce_scalar(result)

            if self.do_final_eval:
                # Collect data
                if isinstance(result, torch.Tensor):
                    data_gpu["result"] = result
                elif isinstance(result, dict):
                    data_gpu.update(result)

                step_data = {key: data_gpu[key] for key in self.metric_keys}
                step_data = transfer_data_to_cpu(step_data)

                for key, value in step_data.items():
                    collect_data[key].append(value.clone())

                if len(self.meta_keys) > 0 and "meta" in data_gpu:
                    for key in self.meta_keys:
                        collect_data[key].append(data_gpu["meta"][key])

            self.after_iter()
        self.after_all_iter()

        if self.do_final_eval:
            # Concat collect_data
            concat_collect_data = OrderedDict()
            for key, tensors in collect_data.items():
                if isinstance(tensors[0], torch.Tensor):
                    concat_collect_data[key] = torch.cat(tensors)
                elif isinstance(tensors[0], list):
                    concat_collect_data[key] = sum(tensors, [])
                else:
                    concat_collect_data[key] = tensors

            # If distributed and use DistributedSampler
            # Gather all collect data to rank 0
            if world_size > 0 and type(val_data_loader.sampler) is torch.utils.data.DistributedSampler:
                concat_collect_data = {key: gather_data(concat_collect_data[key]) for key in
                                       self.metric_keys + self.meta_keys}

            # Do final evaluate
            if rank == 0:
                for metric in self.metrics:
                    self._epoch_outputs[self._mode].update(
                        metric["fn"](*[concat_collect_data[key] for key in metric["keys"]]))

            # Save all data
            if self.save_eval_data and rank == 0:
                # minus 1, means index
                save_path = osp.join(self.work_dir, "eval_{:05d}.pth".format(self.epoch + self.num_folds - 1))
                with FS.get_fs_client(save_path) as client:
                    local_file = client.convert_to_local_path(save_path)
                    torch.save(concat_collect_data, local_file)
                    client.put_object_from_local_file(local_file, save_path)

    def run_epoch(self, data_loaders):
        self.max_epochs = 1
        self.logger.info(f"Begin to evaluate ...")
        self.run_eval_epoch(data_loaders["val"])

    def load_checkpoint(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["state_dict"])

    def save_checkpoint(self) -> dict:
        raise NotImplementedError

    def _build_metrics(self, metric_cfg):
        if isinstance(metric_cfg, (list, tuple)):
            for cfg in metric_cfg:
                self._build_metrics(cfg)
        elif isinstance(metric_cfg, dict):
            fn = METRICS.build(metric_cfg["metric"])
            keys = metric_cfg["keys"]
            self.metrics.append({
                "fn": fn,
                "keys": keys
            })
            self.metric_keys.update(keys)
