#!/usr/bin/python3
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import os
import os.path as osp
import sys
import time
import warnings
from functools import partial

import torch.cuda
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings("ignore")

# For essmc2 python package TODO: install this package via `pip install`
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from essmc2 import Config, DATASETS, MODELS, SOLVERS, get_logger
from essmc2.utils.logger import init_logger
from essmc2.utils.random import set_random_seed
from essmc2.utils.distribute import init_dist, get_dist_info
from essmc2.utils.sampler import MultiFoldDistributedSampler
from essmc2.utils.collate import gpu_batch_collate
from essmc2.utils.ext_module import import_ext_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="""
    Config file path.
    """)
    parser.add_argument("--work_dir", default="./work_dir", type=str, help="""
    Work directory to save log file, checkpoints, config file and tensorboard files.
    """)
    parser.add_argument("--seed", default=123, type=int, help="""
    Random seed.
    """)
    parser.add_argument("--resume_from", type=str, help="""
    Where to resume training task from.
    """)
    parser.add_argument("--dist_launcher", type=str, help="""
    Distribute Launcher, etc 
        pytorch(use pytorch torch.distributed.launch), 
        pai(Platform of Artificial Intelligence in Alibaba), 
        slurm,
        ...
    """)
    parser.add_argument("--dist_backend", default="nccl", type=str, help="""
    Distribute backend, etc
        nccl(default),
        gloo,
        accl(Powered by pai and ais),
        ...
    """)
    parser.add_argument("--local_rank", default=-1, type=int, help="""
    Argument for command torch.distributed.launch or other distribute systems.
    """)
    parser.add_argument("--ext_module", default="papers/CVPR2021-MOSI/impls", type=str, help="""
    Extension module path to be imported for inside custom modules.
    """)
    parser.add_argument("--user_parameters", type=str, help="""
    User script to hack cfg. Make it simpler in a string, such as 'cfg.a=100;cfg.b=200;'
    """)
    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    # Load extension modules
    if args.ext_module is not None:
        import_ext_module(args.ext_module)

    # Load config file
    cfg = Config.load_file(args.config)

    # Load user script to hack cfg object
    # Here you can do anything you want, for example, download compressed dataset and extract it.
    if args.user_parameters is not None:
        try:
            exec(args.user_parameters)
        except Exception as e:
            raise Exception(f"Invoke {args.user_parameters} failed for Reason: {e}")

    # Prepare work directory
    work_dir = args.work_dir
    if not osp.exists(work_dir):
        os.mkdir(work_dir)
    config_name = osp.splitext(osp.basename(args.config))[0]
    work_dir = osp.join(work_dir, config_name)
    cfg.solver["work_dir"] = work_dir

    # Configure distribute environment
    if cfg.get("dist") is None:
        cfg.dist = {}
    distributed = False
    if args.dist_launcher:
        assert args.dist_launcher in ("pytorch", "pai", "slurm")
        cfg.dist["dist_launcher"] = args.dist_launcher
        cfg.dist["dist_backend"] = args.dist_backend

    if cfg.dist.get("dist_launcher") is not None:
        init_dist(backend=cfg.dist["dist_backend"], launcher=args.dist_launcher)
        distributed = True

    rank, world_size = get_dist_info()

    if not osp.exists(work_dir) and rank == 0:
        os.mkdir(work_dir)

    # Configure logger
    run_id = int(time.time())
    log_file = os.path.join(work_dir, f"{run_id}.log")
    logger = get_logger()
    init_logger(logger, log_file, args.dist_launcher)
    logger.info(f"Running task with work directory: {work_dir}")
    logger.info(f"Running task with config: \n{cfg}")
    if args.seed is not None:
        cfg.seed = args.seed

    # Set random seed
    random_seed = cfg.get("seed")
    if random_seed is not None:
        logger.info(f"Set random seed to {random_seed}")
        set_random_seed(random_seed)

    # Load Dataset
    logger.info(f"Building dataset...")
    use_gpu_preprocess = False
    if distributed and cfg.dist["dist_launcher"] == "pytorch":
        # In single-machine, multi cards environment,
        # Check TensorToGPU operation is executed on CORRECT card
        pipeline = cfg.data['train']['dataset']['pipeline']
        for p in pipeline:
            if p["type"] == "TensorToGPU":
                p["device_id"] = rank
                use_gpu_preprocess = True
        if "eval" in cfg.data:
            eval_pipeline = cfg.data["eval"]["dataset"]["pipeline"]
            for p in eval_pipeline:
                if p["type"] == "TensorToGPU":
                    p["device_id"] = rank
    train_dataset = DATASETS.build(cfg.data['train']['dataset'])
    logger.info(f"Built train dataset {train_dataset}")
    if "eval" in cfg.data:
        eval_dataset = DATASETS.build(cfg.data["eval"]["dataset"])
        logger.info(f"Built eval dataset {eval_dataset}")
    else:
        eval_dataset = None

    # Load Dataloader
    pin_memory = cfg.data.get("pin_memory") or False
    if distributed:
        if (cfg.data["train"].get("num_folds") or 1) > 1:
            num_folds = cfg.data["train"].get("num_folds")
            train_sampler = MultiFoldDistributedSampler(train_dataset, num_folds, world_size, rank,
                                                        shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True)
        collate_fn = partial(gpu_batch_collate, device_id=rank) \
            if cfg.dist["dist_launcher"] == "pytorch" and use_gpu_preprocess else None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.data['train']['samples_per_gpu'],
            shuffle=False,
            sampler=train_sampler,
            num_workers=cfg.data['train']['workers_per_gpu'],
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.data['train']['samples_per_gpu'],
            shuffle=True,
            sampler=None,
            num_workers=cfg.data['train']['workers_per_gpu'],
            pin_memory=pin_memory,
        )
    logger.info(f"Built train dataloader {len(train_dataloader)}")

    if eval_dataset is not None:
        if distributed:
            eval_sampler = DistributedSampler(eval_dataset, world_size, rank, shuffle=False)
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=cfg.data['eval']['samples_per_gpu'],
                shuffle=False,
                sampler=eval_sampler,
                num_workers=cfg.data['eval']['workers_per_gpu'],
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=cfg.data['eval']['samples_per_gpu'],
                shuffle=False,
                sampler=None,
                num_workers=cfg.data['eval']['workers_per_gpu'],
                pin_memory=pin_memory
            )
        logger.info(f"Built eval dataloader {len(eval_dataloader)}")

    else:
        eval_dataloader = None

    data = dict(train=train_dataloader)
    if eval_dataloader:
        data["val"] = eval_dataloader

    # Load Model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)
    if distributed and cfg.dist.get("sync_bn") is True:
        logger.info("Convert BatchNorm to Synchronized BatchNorm...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    if distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model)
    logger.info(f"Built model: \n{model}")

    # Load Solver
    if args.resume_from is not None:
        cfg.solver["resume_from"] = args.resume_from
    logger.info("Building solver...")
    solver = SOLVERS.build(model, cfg.solver)
    logger.info(f"Built solver: {solver}")

    # Save config
    if rank == 0:
        config_path = osp.join(work_dir, "final_" + osp.basename(args.config))
        cfg.dump(config_path)

    # Begin solve
    solver.solve(data)
    logger.info(f"Solved")
