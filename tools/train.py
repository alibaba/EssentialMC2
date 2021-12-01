#!/usr/bin/python3
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import os
import os.path as osp
import time
from functools import partial

import torch.cuda
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from essmc2 import Config, DATASETS, MODELS, SOLVERS, get_logger
from essmc2.utils.collate import gpu_batch_collate
from essmc2.utils.distribute import init_dist, get_dist_info
from essmc2.utils.ext_module import import_ext_module
from essmc2.utils.file_systems import FS, LocalFs
from essmc2.utils.logger import init_logger
from essmc2.utils.random import set_random_seed
from essmc2.utils.sampler import MultiFoldDistributedSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--work_dir", default="./work_dir", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--data_root_dir", type=str)
    parser.add_argument("--annotation_dir", type=str)
    parser.add_argument("--backbone_pretrain", type=str)
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
    parser.add_argument("--ext_module", default="", type=str, help="""
    Extension module path to be imported for inside custom modules.
    """)
    parser.add_argument("--user_parameters", type=str, help="""
    User script to hack cfg. Make it simpler in a string, such as 'cfg.a=100;cfg.b=200;'
    """)
    return parser.parse_args()


def get_model(cfg, logger):
    model = MODELS.build(cfg.model)
    if cfg.dist.distributed and cfg.dist.get("sync_bn") is True:
        logger.info("Convert BatchNorm to Synchronized BatchNorm...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    if cfg.dist.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()])
    return model


def get_data(cfg, logger):
    use_gpu_preprocess = False
    rank, world_size = get_dist_info()
    if cfg.dist.distributed and cfg.dist["dist_launcher"] == "pytorch":
        # when use single-machine, multi cards,
        # check TensorToGPU operation on CORRECT card
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
    train_dataset = DATASETS.build(cfg.data['train']['dataset'], fs_cfg=cfg.get("file_systems"))
    logger.info(f"Built train dataset {train_dataset}")
    if "eval" in cfg.data:
        eval_dataset = DATASETS.build(cfg.data["eval"]["dataset"], fs_cfg=cfg.get("file_systems"))
        logger.info(f"Built eval dataset {eval_dataset}")
    else:
        eval_dataset = None

    # Load Dataloader
    pin_memory = cfg.data.get("pin_memory") or False
    if cfg.dist.distributed:
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
    logger.info(f"Built train dataloader with len {len(train_dataloader)}")

    if eval_dataset is not None:
        if cfg.dist.distributed:
            eval_sampler = DistributedSampler(eval_dataset, world_size, rank, shuffle=False)
            collate_fn = partial(gpu_batch_collate, device_id=rank) \
                if cfg.dist["dist_launcher"] == "pytorch" and use_gpu_preprocess else None
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=cfg.data['eval']['samples_per_gpu'],
                shuffle=False,
                sampler=eval_sampler,
                num_workers=cfg.data['eval']['workers_per_gpu'],
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collate_fn
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=cfg.data['eval']['samples_per_gpu'],
                shuffle=False,
                sampler=None,
                num_workers=cfg.data['eval']['workers_per_gpu'],
                pin_memory=pin_memory,
                drop_last=False
            )

        logger.info(f"Built eval dataloader with len {len(eval_dataloader)}")

    else:
        eval_dataloader = None

    data = dict(train=train_dataloader)
    if eval_dataloader:
        data["val"] = eval_dataloader

    return data


def main():
    args = parse_args()

    # Load extension modules
    if args.ext_module is not None:
        import_ext_module(args.ext_module)

    # Load config file
    cfg = Config.load_file(args.config)

    # Load user script to modify cfg
    if args.user_parameters is not None:
        try:
            exec(args.user_parameters)
        except Exception as e:
            raise Exception(f"Invoke {args.user_parameters} failed for Reason: {e}")

    # ------ Change config by args and specify task ----- #
    # # Change distribute config
    if cfg.get("dist") is None:
        cfg.dist = dict(distributed=False)
    if args.dist_launcher:
        assert args.dist_launcher in ("pytorch", "pai", "slurm")
        cfg.dist["dist_launcher"] = args.dist_launcher
        cfg.dist["dist_backend"] = args.dist_backend
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
    if args.backbone_pretrain is not None:
        cfg.model["use_pretrain"] = True
        cfg.model["load_from"] = args.backbone_pretrain
    # # Datasets
    if args.data_root_dir is not None:
        data_root_dir = args.data_root_dir
        annotation_dir = args.annotation_dir
        cfg.data["train"]["dataset"]["data_root_dir"] = data_root_dir
        cfg.data["train"]["dataset"]["annotation_dir"] = annotation_dir
        if "eval" in cfg.data:
            cfg.data["eval"]["dataset"]["data_root_dir"] = data_root_dir
            cfg.data["eval"]["dataset"]["annotation_dir"] = annotation_dir
        if "test" in cfg.data:
            cfg.data["test"]["dataset"]["data_root_dir"] = data_root_dir
            cfg.data["test"]["dataset"]["annotation_dir"] = annotation_dir
    # # Resume
    if args.resume_from is not None:
        cfg.solver["resume_from"] = args.resume_from
    # ------ Done Change config by args and specify task ----- #

    # Configure file system client
    FS.init_fs_client(cfg.get("file_systems"))

    # Configure distribute environment
    if cfg.dist.get("dist_launcher") is not None:
        init_dist(backend=cfg.dist["dist_backend"], launcher=args.dist_launcher)
    rank, world_size = get_dist_info()

    # Prepare work directory
    work_fs_client = FS.get_fs_client(work_dir)
    if type(work_fs_client) is LocalFs:
        if not osp.exists(work_dir) and rank == 0:
            os.makedirs(work_dir, exist_ok=True)
    else:
        local_work_dir = osp.join("./", osp.basename(args.work_dir), config_name)
        os.makedirs(local_work_dir, exist_ok=True)
        work_fs_client.add_target_local_map(work_dir, local_work_dir)

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
    torch.multiprocessing.set_start_method('spawn')

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

    # Save config
    if rank == 0:
        config_path = osp.join(work_dir, "final_" + osp.basename(args.config))
        local_config_path = work_fs_client.convert_to_local_path(config_path)
        cfg.dump(local_config_path)
        work_fs_client.put_object_from_local_file(local_config_path, config_path)

    # Begin solve
    solver.solve(data)
    logger.info(f"Solved")


if __name__ == "__main__":
    main()
