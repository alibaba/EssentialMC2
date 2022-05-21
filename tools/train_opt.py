#!/usr/bin/python3
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import datetime
import os
import os.path as osp
import sys
import time

import torch.cuda

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))

from essmc2 import Config, SOLVERS, get_logger
from essmc2.utils.distribute import init_dist, get_dist_info
from essmc2.utils.ext_module import import_ext_module
from essmc2.utils.file_systems import FS
from essmc2.utils.logger import init_logger
from essmc2.utils.random import set_random_seed

from essmc2.apis.config import get_train_base_config
from essmc2.apis.model import get_model
from essmc2.apis.data import get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="""
    path to config file, must be a local file
    """)
    parser.add_argument("--work_dir", default="./work_dir", type=str, help="""
    directory to save outputs, default is ./work_dir
    """)
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--resume_from", type=str, help="""
    path to checkpoint for resuming training, default is None
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


def _modify_cfg_by_args(cfg, args):
    # # Change distribute config
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
    # # Seed
    if args.seed is not None:
        cfg.seed = args.seed
    # # Model
    if args.pretrain is not None:
        cfg.model.pretrain = args.pretrain
    # # Datasets
    if args.data_root_dir is not None:
        data_root_dir = args.data_root_dir
        annotation_dir = args.annotation_dir
        cfg.data.train.dataset.data_root_dir = data_root_dir
        cfg.data.train.dataset.annotation_dir = annotation_dir
        if "eval" in cfg.data:
            cfg.data.eval.dataset.data_root_dir = data_root_dir
            cfg.data.eval.dataset.annotation_dir = annotation_dir
    # # Resume
    if args.resume_from is not None:
        cfg.solver.resume_from = args.resume_from

    return cfg


def main():
    args = parse_args()

    # Load extension modules
    if args.ext_module is not None:
        import_ext_module(args.ext_module)

    # Load config file
    cfg = get_train_base_config()
    ext_cfg = Config.load(args.config)
    cfg = Config.merge_a_into_b(ext_cfg, cfg)

    # Change config by args and specify task
    cfg = _modify_cfg_by_args(cfg, args)

    # Load user script to modify cfg
    if args.user_parameters is not None:
        try:
            exec(args.user_parameters)
        except Exception as e:
            raise Exception(f"Invoke {args.user_parameters} failed for Reason: {e}")

    # TODO: cfg should be frozen from now on

    work_dir = cfg.solver.work_dir
    config_name = osp.splitext(osp.basename(args.config))[0]

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
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(work_dir, f"train-{run_id}.log")
    logger = get_logger()
    init_logger(logger, log_file, cfg.dist.launcher)
    logger.info(f"Running task with work directory: {work_dir}")
    logger.info(f"Running task with config: \n{cfg}")

    # Set random seed
    random_seed = cfg.get("seed")
    if random_seed is not None:
        logger.info(f"Set random seed to {random_seed}")
        set_random_seed(random_seed)

    # Set cudnn configs
    if cfg.cudnn.deterministic is not None:
        torch.backends.cudnn.deterministic = cfg.cudnn.deterministic
    if cfg.cudnn.benchmark is not None:
        torch.backends.cudnn.benchmark = cfg.cudnn.benchmark

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
    solver = SOLVERS.build(model, cfg.solver, logger=logger)
    logger.info(f"Built solver: {solver}")

    # Save final config
    if rank == 0:
        config_path = osp.join(work_dir, "final_" + osp.basename(args.config))
        with FS.put_to(config_path) as local_config_path:
            cfg.dump(local_config_path)

    # Begin solve
    solver.solve(data)
    logger.info(f"Solved")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
