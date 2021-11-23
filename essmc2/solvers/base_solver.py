# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from abc import abstractmethod, ABCMeta
from collections import OrderedDict

import torch

from essmc2.hooks import HOOKS
from essmc2.utils.logger import get_logger
from ..lr_schedulers import LR_SCHEDULERS
from ..optimizers import OPTIMIZERS
from essmc2.utils.typing import check_dict_of_str_dict


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
        self.optimizer: torch.optim.optimizer.Optimizer = self._get_optimizer(optimizer)
        self.lr_scheduler = self._get_lr_scheduler(lr_scheduler)
        self.resume_from = resume_from
        self.work_dir = work_dir
        self.logger = logger or get_logger()
        self.envs = envs or {}
        self.max_epochs = max_epochs
        self.num_folds = num_folds
        self._epoch = 0
        self._epoch_max_iter = 0
        self._iter = 0
        self._total_train_iter = 0
        self._total_eval_iter = 0
        self._total_test_iter = 0
        self._iter_outputs = dict()
        self._epoch_outputs = dict()
        self._mode = 'train'  # current mode, 'train' or 'val'
        self._hooks = []
        self._load_hook(hooks)
        self.data_loaders = {}

    def solve(self, data_loaders):
        # get a reference to data for hooks to use
        self.data_loaders = data_loaders
        self.before_solve()
        while self._epoch < self.max_epochs:
            self.logger.info(f"Begin to solve epoch [{self._epoch}/{self.max_epochs}]...")
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

    def after_epoch(self, *args, **kwargs):
        [t.after_epoch(self) for t in self._hooks]
        self._epoch += self.num_folds
        self._iter = 0

    def before_all_iter(self):
        [t.before_all_iter(self) for t in self._hooks]

    def before_iter(self, *args, **kwargs):
        [t.before_iter(self) for t in self._hooks]

    def after_iter(self, *args, **kwargs):
        [t.after_iter(self) for t in self._hooks]
        if self.is_train_mode:
            self._total_train_iter += 1
        elif self.is_eval_mode:
            self._total_eval_iter += 1
        else:
            self._total_test_iter += 1

        self._iter += 1

    def after_all_iter(self, *args, **kwargs):
        [t.after_all_iter(self) for t in self._hooks]

    def collect_log_vars(self) -> OrderedDict:
        ret = OrderedDict()
        if self.is_train_mode and self.optimizer is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            ret["lr"] = lr
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
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = new_epoch

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, new_iter):
        self._iter = new_iter

    @property
    def total_train_iter(self):
        return self._total_train_iter

    @property
    def total_eval_iter(self):
        return self._total_eval_iter

    @property
    def total_test_iter(self):
        return self._total_test_iter

    @property
    def total_iter(self):
        if self.mode == "train":
            return self.total_train_iter
        elif self.mode == "eval":
            return self.total_eval_iter
        else:
            return self.total_test_iter

    @property
    def epoch_max_iter(self):
        return self._epoch_max_iter

    @property
    def mode(self):
        return self._mode

    @property
    def iter_outputs(self):
        return self._iter_outputs

    @property
    def epoch_outputs(self):
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
                self._hooks.append(HOOKS.build(hook_cfg))
        self._hooks.sort(key=lambda a: a.priority)

    def get_optim_parameters(self):
        return self.model.parameters()

    def _get_optimizer(self, cfg):
        if cfg is None:
            return None
        return OPTIMIZERS.build(self.get_optim_parameters(), cfg)

    def _get_lr_scheduler(self, cfg):
        if cfg is None:
            return None
        if self.optimizer is None:
            return None
        return LR_SCHEDULERS.build(self.optimizer, cfg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
