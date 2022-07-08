# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import abstractmethod, ABCMeta
from collections import OrderedDict, defaultdict

import torch

from essmc2.hooks import HOOKS
from essmc2.utils.distribute import dist
from essmc2.utils.logger import get_logger
from essmc2.utils.typing import check_dict_of_str_dict
from ..lr_schedulers import LR_SCHEDULERS
from ..optimizers import OPTIMIZERS
from typing import Union, Dict


class BaseSolver(object, metaclass=ABCMeta):
    """ Base Solver.

    Args:
        model (torch.nn.Module): Model to train or eval.
        optimizer (torch.optim.Optimizer, None): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, None): Learning rate scheduler.
        resume_from (str, None): Checkpoint path to resume.
        work_dir (str, None): Work directory where to save checkpoint, tensorboard logs or other things.
        logger (logging.Logger, None): If None, use global logger.
        envs (dict, None): Environment constants or some hyper params.
        max_epochs (int): Max training or inference epoch numbers, default is 1.
        num_folds (int): Number of training dataset fold numbers, will affect log or lr scheduler.
        hooks (List[Hook], None): List of hook configurations.
    """

    def __init__(self,
                 model,
                 optimizer=None,
                 lr_scheduler=None,
                 resume_from=None,
                 work_dir=None,
                 logger=None,
                 envs=None,
                 max_epochs=1,
                 num_folds=1,
                 hooks=None):
        self.model: torch.nn.Module = model
        self.optimizer: Union[torch.optim.optimizer.Optimizer, Dict[str, torch.optim.optimizer.Optimizer]] = self.build_optimizer(optimizer)
        self.lr_scheduler = self.build_lr_scheduler(lr_scheduler)
        self.resume_from = resume_from
        self.work_dir = work_dir
        self.logger = logger or get_logger()
        self.envs = envs or {}
        self.max_epochs = max_epochs
        self._num_folds = num_folds
        self._epoch: int = 0
        # epoch_max_iter, iter, total_iter, iter_outputs, epoch_outputs
        # values is different according to self._mode
        self._epoch_max_iter: defaultdict = defaultdict(int)
        self._iter: defaultdict = defaultdict(int)
        self._total_iter: defaultdict = defaultdict(int)
        self._iter_inputs = defaultdict(dict)
        self._iter_outputs = defaultdict(dict)
        self._agg_iter_outputs = defaultdict(dict)
        self._epoch_outputs = defaultdict(dict)
        self._mode: str = 'train'  # current mode, 'train' 'eval' 'test' ...
        self._hooks = []
        self._load_hook(hooks)
        self.data_loaders = {}
        self.loss = None  # loss tensor

    def solve(self, data_loaders):
        # get a reference to data for hooks to use
        self.data_loaders = data_loaders
        self.before_solve()
        while self._epoch < self.max_epochs:
            self.logger.info(f"Begin to solve at Epoch [{self._epoch}/{self.max_epochs}]...")
            self.before_epoch()
            self.run_epoch(data_loaders)
            self.after_epoch()
        self.after_solve()

    def before_solve(self, *args, **kwargs):
        [t.before_solve(self) for t in self._hooks]

    def after_solve(self, *args, **kwargs):
        [t.after_solve(self) for t in self._hooks]

    @abstractmethod
    def run_epoch(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        [t.before_epoch(self) for t in self._hooks]

    def before_all_iter(self):
        [t.before_all_iter(self) for t in self._hooks]

    def before_iter(self, *args, **kwargs):
        [t.before_iter(self) for t in self._hooks]

    def after_iter(self, *args, **kwargs):
        [t.after_iter(self) for t in self._hooks]
        self._total_iter[self._mode] += 1
        self._iter[self._mode] += 1

    def after_all_iter(self, *args, **kwargs):
        [t.after_all_iter(self) for t in self._hooks]

    def after_epoch(self, *args, **kwargs):
        [t.after_epoch(self) for t in self._hooks]
        self._epoch += self._num_folds
        self._iter.clear()
        self._iter_inputs.clear()
        self._iter_outputs.clear()
        self._epoch_outputs.clear()

    def collect_log_vars(self) -> OrderedDict:
        ret = OrderedDict()
        if self.is_train_mode and self.optimizer is not None:
            if not isinstance(self.optimizer, dict):
                lr = self.optimizer.param_groups[0]["lr"]
                ret["lr"] = lr
            else:
                for key, value in self.optimizer.items():
                    lr = self.optimizer[key].param_groups[0]['lr']
                    ret[f'{key}_lr'] = lr
        return ret

    @abstractmethod
    def load_checkpoint(self, checkpoint: dict):
        """
        Load checkpoint function
        :param checkpoint: all tensors are on cpu, you need to transfer to gpu by hand
        :return:
        """
        pass

    @abstractmethod
    def save_checkpoint(self) -> dict:
        """
        Save checkpoint function, you need to transfer all tensors to cpu by hand
        :return:
        """
        pass

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = new_epoch

    @property
    def iter(self) -> int:
        return self._iter[self._mode]

    @property
    def total_iter(self) -> int:
        return self._total_iter[self._mode]

    @property
    def epoch_max_iter(self) -> int:
        return self._epoch_max_iter[self._mode]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def iter_inputs(self) -> dict:
        return self._iter_inputs[self._mode]

    @property
    def iter_outputs(self) -> dict:
        return self._iter_outputs[self._mode]

    @property
    def agg_iter_outputs(self) -> dict:
        return self._agg_iter_outputs

    @agg_iter_outputs.setter
    def agg_iter_outputs(self, new_outputs):
        assert type(new_outputs) is dict
        self._agg_iter_outputs[self._mode] = new_outputs

    @property
    def epoch_outputs(self) -> dict:
        return self._epoch_outputs

    @property
    def is_train_mode(self):
        return self._mode == "train"

    @property
    def is_eval_mode(self):
        return self._mode == "eval"

    @property
    def is_test_mode(self):
        return self._mode == "test"

    def train_mode(self):
        self.model.train()
        self._mode = "train"

    def eval_mode(self):
        self.model.eval()
        self._mode = "eval"

    def test_mode(self):
        self.model.eval()
        self._mode = "test"

    def _load_hook(self, hooks):
        if hooks is not None and len(hooks) > 0:
            if check_dict_of_str_dict(hooks, contains_type=True):
                hooks = list(hooks.values())
            for hook_cfg in hooks:
                if hook_cfg is None:
                    continue
                self._hooks.append(HOOKS.build(hook_cfg))
        self._hooks.sort(key=lambda a: a.priority)

    def build_optimizer(self, cfg):
        if cfg is None:
            return None
        return OPTIMIZERS.build(self.model, cfg)

    def build_lr_scheduler(self, cfg):
        if cfg is None:
            return None
        if self.optimizer is None:
            return None
        return LR_SCHEDULERS.build(self.optimizer, cfg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def _reduce_scalar(self, data_dict: dict):
        """ Only reduce all scalar tensor values if distributed.
        Any way, loss tensor will be specially processed just in case.

        Args:
            data_dict: Dict result returned by model.

        Returns:
            A new data dict whose tensor scalar values is all-reduced.

        """
        if "loss" in data_dict:
            self.loss = data_dict["loss"]
            data_dict["loss"] = self.loss.data.clone()

        if isinstance(data_dict, OrderedDict):
            keys = data_dict.keys()
        else:
            keys = sorted(list(data_dict.keys()))

        ret = OrderedDict()
        for key in keys:
            value = data_dict[key]
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                if dist.is_available() and dist.is_initialized():
                    value = value.data.clone()
                    dist.all_reduce(value.div_(dist.get_world_size()))
                ret[key] = value
            else:
                ret[key] = value

        return ret
