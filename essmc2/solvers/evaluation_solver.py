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
from ..utils.sampler import EvalDistributedSampler


def _get_value(data: dict, key: str):
    """ Recursively get value from data by a multi-level key.

    Args:
        data (dict):
        key (str): 'data', 'meta.path', 'a.b.c'

    Returns:
        Value.

    """
    if not isinstance(data, dict):
        return None
    if key in data:
        return data[key]
    elif "." in key:
        par_key = key.split(".")[0]
        sub_key = ".".join(key.split(".")[1:])
        if par_key in data:
            return _get_value(data[par_key], sub_key)
    return None


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
            Either do_final_eval or save_eval_data is True, do collect action.
        save_eval_data (bool): If True, save all collected data. Default path is
            "WORK_DIR/eval_{epoch}_data.pth".
            Either do_final_eval or save_eval_data is True, do collect action.
        eval_metric_cfg (dict, Sequence, optional): Metric function descriptor like
            {
                "metric": dict(type="accuracy", topk=(1, )),
                "keys": ("result", "gt_label")
            }
        extra_keys (tuple or list, optional): Extra keys to collect, e.g. ["meta.image_path"]
    """

    def __init__(self,
                 model,
                 eval_interval=1,
                 do_final_eval=False,
                 save_eval_data=False,
                 eval_metric_cfg=None,
                 extra_keys=None,
                 **kwargs):
        super().__init__(model, **kwargs)

        self.eval_interval = eval_interval
        self.do_final_eval = do_final_eval
        self.save_eval_data = save_eval_data
        self.metrics = []
        self._collect_keys = set()
        if self.do_final_eval or self.save_eval_data:
            self._build_metrics(eval_metric_cfg)
            self._collect_keys.update(list(extra_keys or []))
            self._collect_keys = sorted(list(self._collect_keys))
            if len(self._collect_keys) > 0:
                self.logger.info(f"{', '.join(self._collect_keys)} will be collected during eval epoch")

    @torch.no_grad()
    def run_eval_epoch(self, val_data_loader):
        collect_data = defaultdict(list)

        rank, world_size = get_dist_info()

        # Enter evaluate mode
        self.eval_mode()
        self._epoch_max_iter[self._mode] = len(val_data_loader)
        self.before_all_iter()
        for data in val_data_loader:
            self.before_iter()
            self._iter_inputs[self._mode] = transfer_data_to_cuda(data)
            result = self.model(**self._iter_inputs[self._mode])

            self._iter_outputs[self._mode] = self._reduce_scalar(result)

            if self.do_final_eval or self.save_eval_data:
                # Collect data
                data_gpu = self._iter_inputs[self._mode].copy()
                if isinstance(result, torch.Tensor):
                    data_gpu["result"] = result
                elif isinstance(result, dict):
                    data_gpu.update(result)

                step_data = OrderedDict()
                for key in self._collect_keys:
                    value = _get_value(data_gpu, key)
                    if value is None:
                        raise ValueError(f"Cannot get valid value from model input or output data with key {key}")
                    step_data[key] = value

                step_data = transfer_data_to_cpu(step_data)

                for key, value in step_data.items():
                    if isinstance(value, torch.Tensor):
                        collect_data[key].append(value.clone())
                    else:
                        collect_data[key].append(value)

            self.after_iter()
        self.after_all_iter()

        if self.do_final_eval or self.save_eval_data:
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
            if world_size > 0 and type(val_data_loader.sampler) is EvalDistributedSampler:
                concat_collect_data = {key: gather_data(concat_collect_data[key]) for key in self._collect_keys}

            # Do final evaluate
            if self.do_final_eval and rank == 0:
                for metric in self.metrics:
                    self._epoch_outputs[self._mode].update(
                        metric["fn"](*[concat_collect_data[key] for key in metric["keys"]]))

            # Save all data
            if self.save_eval_data and rank == 0:
                # minus 1, means index
                save_path = osp.join(self.work_dir, "eval_{:05d}.pth".format(self.epoch + self.num_folds))
                with FS.put_to(save_path) as local_file:
                    torch.save(concat_collect_data, local_file)

    def run_epoch(self, data_loaders):
        self.max_epochs = 1
        self.logger.info(f"Begin to evaluate ...")
        self.run_eval_epoch(data_loaders["eval"])

    def load_checkpoint(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["state_dict"])

    def save_checkpoint(self) -> dict:
        return {}

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
            self._collect_keys.update(keys)

