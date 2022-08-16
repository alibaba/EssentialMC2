import logging
from functools import partial
from typing import Optional

import torch.cuda
from torch.utils.data import DistributedSampler, DataLoader

from essmc2.datasets import DATASETS
from essmc2.utils.collate import gpu_batch_collate
from essmc2.utils.config import Config
from essmc2.utils.data import worker_init_fn
from essmc2.utils.distribute import get_dist_info
from essmc2.utils.logger import get_logger
from essmc2.utils.sampler import MultiFoldDistributedSampler, EvalDistributedSampler, MultiFoldRandomSampler


def get_data(cfg: Config, logger: Optional[logging.Logger] = None, eval_only: bool = False):
    logger = logger or get_logger()
    device_id = torch.cuda.current_device()
    data = {}
    if not eval_only:
        if "train" in cfg.data:
            train_dataloader = _get_train_data(cfg, device_id, logger)
            data['train'] = train_dataloader
    if cfg.data.get("eval") is not None:
        eval_dataloader = _get_eval_data(cfg, device_id, logger)
        data['eval'] = eval_dataloader

    return data


def _get_train_data(cfg: Config, device_id: int, logger: logging.Logger):
    rank, world_size = get_dist_info()

    train_dataset = DATASETS.build(cfg.data.train.dataset)
    logger.info(f'Built train dataset {train_dataset}')

    pin_memory = cfg.data.get("pin_memory") or False
    collate_fn = partial(gpu_batch_collate, device_id=device_id)
    train_worker_init_fn = partial(worker_init_fn,
                                   seed=cfg.seed,
                                   worker_device=f'cuda:{device_id}',
                                   file_systems=cfg.get("file_systems"))
    if cfg.dist.distributed:
        shuffle = False
        if (cfg.solver.get("num_folds") or 1) > 1:
            num_folds = cfg.solver.get("num_folds")
            train_sampler = MultiFoldDistributedSampler(train_dataset, num_folds, world_size, rank, shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True)
        drop_last = True
    else:
        num_folds = cfg.solver.get("num_folds") or 1
        if num_folds == 1:
            shuffle = True
            train_sampler = None
            drop_last = False
        else:
            shuffle = False
            train_sampler = MultiFoldRandomSampler(train_dataset, num_folds)
            drop_last = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train.samples_per_gpu,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=cfg.data.train.workers_per_gpu,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=train_worker_init_fn
    )

    logger.info(f"Built train dataloader with len {len(train_dataloader)}")

    return train_dataloader


def _get_eval_data(cfg: Config, device_id=0, logger: Optional[logging.Logger] = None):
    rank, world_size = get_dist_info()

    eval_dataset = DATASETS.build(cfg.data.eval.dataset)
    logger.info(f'Built eval dataset {eval_dataset}')

    pin_memory = cfg.data.get("pin_memory") or False
    collate_fn = partial(gpu_batch_collate, device_id=device_id)
    eval_worker_init_fn = partial(worker_init_fn,
                                  seed=cfg.seed,
                                  worker_device=f'cuda:{device_id}',
                                  file_systems=cfg.get("file_systems"))
    if cfg.dist.distributed:
        eval_sampler = EvalDistributedSampler(eval_dataset, world_size, rank)
    else:
        eval_sampler = None

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.data.eval.samples_per_gpu,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=cfg.data.eval.workers_per_gpu,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=eval_worker_init_fn
    )

    logger.info(f"Built eval dataloader with len {len(eval_dataloader)}")

    return eval_dataloader
