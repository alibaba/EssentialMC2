#!/usr/bin/python3
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import os
import os.path as osp
import time
from functools import partial

import torch.cuda
from torch.utils.data import DataLoader

from essmc2 import Config, DATASETS, MODELS, SOLVERS, get_logger
from essmc2.utils.collate import gpu_batch_collate
from essmc2.utils.data import worker_init_fn
from essmc2.utils.distribute import init_dist, get_dist_info
from essmc2.utils.ext_module import import_ext_module
from essmc2.utils.file_systems import FS
from essmc2.utils.logger import init_logger
from essmc2.utils.random import set_random_seed
from essmc2.utils.sampler import EvalDistributedSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="""
    path to config file, must be a local file
    """)
    parser.add_argument("--work_dir", default="./work_dir", type=str, help="""
    directory to save outputs, default is ./work_dir
    """)
    parser.add_argument("--data_root_dir", type=str, help="""
    root directory to load image, video or text, default is None
    """)
    parser.add_argument("--annotation_dir", type=str, help="""
    directory to load annotations, default is None
    """)
    parser.add_argument("--pretrain", type=str, help="""
    path to pretrained model, default is None
    """)
    parser.add_argument("--dist_launcher", type=str, help="""
    distribute launcher, etc 
        pytorch(use pytorch torch.distributed.launch), 
        pai(Platform of Artificial Intelligence in Alibaba), 
        slurm,
        ...
    default is None
    """)
    parser.add_argument("--dist_backend", default="nccl", type=str, help="""
    distribute backend, etc
        nccl(default),
        gloo,
        ...
    default is None
    """)
    parser.add_argument("--local_rank", default=-1, type=int, help="""
    argument for command torch.distributed.launch or other distribute systems, default is -1
    """)
    parser.add_argument("--ext_module", default="", type=str, help="""
    extension module path to be imported for custom modules, default is empty
    """)
    parser.add_argument("--user_parameters", type=str, help="""
    user's python script to hack cfg, make it simpler in one line python statement, such as 'cfg.a=100;cfg.b=200', default is None
    """)
    return parser.parse_args()


def get_model(cfg, logger):
    model = MODELS.build(cfg.model)

    model = model.cuda()
    return model


def get_data(cfg, logger):
    use_pytorch_launcher = cfg.dist.distributed and cfg.dist.launcher == 'pytorch'
    rank, world_size = get_dist_info()

    eval_dataset = DATASETS.build(cfg.data["eval"]["dataset"])
    logger.info(f"Built eval dataset {eval_dataset}")

    # Load Dataloader
    pin_memory = cfg.data.get("pin_memory") or False
    collate_fn = partial(gpu_batch_collate, device_id=rank if use_pytorch_launcher else 0)
    eval_worker_init_fn = partial(worker_init_fn, file_systems=cfg.get('file_systems'))
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

    data = dict(eval=eval_dataloader)

    return data


def main():
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    # Load extension modules
    if args.ext_module is not None:
        import_ext_module(args.ext_module)

    # Load config file
    cfg = Config.load(args.config)

    # Load user script to modify cfg
    if args.user_parameters is not None:
        try:
            exec(args.user_parameters)
        except Exception as e:
            raise Exception(f"Invoke {args.user_parameters} failed for Reason: {e}")

    # ------ Change config by args and specify task ----- #
    # # Change distribute config
    if cfg.get("dist") is None:
        cfg.dist = dict(distributed=False, launcher=None, backend=None)
    if args.dist_launcher:
        assert args.dist_launcher in ("pytorch", "pai", "slurm")
        cfg.dist.launcher = args.dist_launcher
        cfg.dist.backend = args.dist_backend
        cfg.dist.distributed = True
    # # Change work directory
    work_dir = args.work_dir
    config_name = osp.splitext(osp.basename(args.config))[0]
    work_dir = osp.join(work_dir, config_name)
    cfg.solver["work_dir"] = work_dir
    # # Model
    if args.pretrain is not None:
        cfg.model.pretrain = args.pretrain
    # # Datasets
    if args.data_root_dir is not None:
        cfg.data.eval.dataset.data_root_dir = args.data_root_dir
        cfg.data.eval.dataset.annotation_dir = args.annotation_dir
    # ------ Done Change config by args and specify task ----- #

    # Configure file system client
    FS.init_fs_client(cfg.get("file_systems"))

    # Configure distribute environment
    if cfg.dist.launcher is not None:
        init_dist(backend=cfg.dist.backend, launcher=cfg.dist.launcher)
    rank, world_size = get_dist_info()

    # Prepare work directory
    if rank == 0:
        local_work_dir = work_dir if FS.is_local_client(work_dir) \
            else osp.join("./", osp.basename(args.work_dir), config_name)
        os.makedirs(local_work_dir, exist_ok=True)
        FS.add_target_local_map(work_dir, local_work_dir)

    # Configure logger
    run_id = int(time.time())
    log_file = os.path.join(work_dir, f"{run_id}.log")
    logger = get_logger()
    init_logger(logger, log_file, args.dist_launcher)
    logger.info(f"Running task with work directory: {work_dir}")
    logger.info(f"Running task with config: \n{cfg}")

    # Set torch constant
    random_seed = cfg.get("seed")
    if random_seed is not None:
        logger.info(f"Set random seed to {random_seed}")
        set_random_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load Model
    logger.info("Building model...")
    model = get_model(cfg, logger)
    logger.info(f"Built model: \n{model}")

    # Load Dataset
    logger.info(f"Building data...")
    data = get_data(cfg, logger)
    logger.info(f"Built data: {list(data.keys())}")

    # Load Solver
    logger.info("Building solver...")
    cfg.solver.type = "EvaluationSolver"
    solver = SOLVERS.build(model, cfg.solver, logger=logger)
    logger.info(f"Built solver: {solver}")

    # Save config
    if rank == 0:
        config_path = osp.join(work_dir, "final_" + osp.basename(args.config))
        with FS.put_to(config_path) as local_config_path:
            cfg.dump(local_config_path)

    # Begin solve
    solver.solve(data)
    logger.info(f"Solved")


if __name__ == "__main__":
    main()
