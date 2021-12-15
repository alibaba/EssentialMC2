# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import argparse
import copy
import os
import os.path as osp
import time

from torch.utils.data import DataLoader

from essmc2 import Config, DATASETS, MODELS, SOLVERS, get_logger
from essmc2.utils.ext_module import import_ext_module
from essmc2.utils.logger import init_logger
from essmc2.utils.random import set_random_seed

# Import NGC impls
import_ext_module("./impls")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--work_dir", default="./work_dir", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--resume_from", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.load_file(args.config)
    work_dir = args.work_dir
    if not osp.exists(work_dir):
        os.mkdir(work_dir)
    config_name = osp.splitext(osp.basename(args.config))[0]
    work_dir = osp.join(work_dir, config_name)

    cfg.solver["work_dir"] = work_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.resume_from is not None:
        cfg.solver["resume_from"] = args.resume_from

    run_id = int(time.time())
    if not osp.exists(work_dir):
        os.mkdir(work_dir)
    log_file = os.path.join(work_dir, f"{run_id}.log")
    logger = get_logger()
    init_logger(logger, log_file)
    logger.info(f"Running task with work directory: {work_dir}")
    logger.info(f"Running task with log file: {log_file}")
    logger.info(f"Running task with config: \n{cfg}")
    random_seed = cfg.get("seed")
    if random_seed is not None:
        logger.info(f"Set random seed to {random_seed}")
        set_random_seed(random_seed)

    # Load Dataset, DataLoader, etc
    logger.info(f"Building dataset...")

    train_dataset = DATASETS.build(cfg.data['train']['dataset'])
    logger.info(f"Built train dataset {train_dataset}")

    # when use ood noise, carefully check train and test dataset will not use same openset instances
    if cfg.solver.hyper_params["openset"] and cfg.data['test']['dataset']['type'] == "CifarNoisyOpensetDataset":
        cfg.data['test']['dataset']['train_used_idx'] = train_dataset.openset_select_ids
    test_dataset = DATASETS.build(cfg.data['test']['dataset'])
    logger.info(f"Built test dataset {test_dataset}")

    # In this task, eval dataset should be EXACT SAME as train dataset, especially noise generators
    # So here we just copy train dataset as eval dataset
    eval_dataset = copy.copy(train_dataset)
    eval_dataset.mode = "eval"
    eval_dataset.pipeline = test_dataset.pipeline
    logger.info(f"Built eval dataset {eval_dataset}")

    if cfg.solver.hyper_params["dataset_name"] == "webvision" and "imagenet" in cfg.data:
        imagenet_dataset = DATASETS.build(cfg.data["imagenet"]["dataset"])
        logger.info(f"Built imagenet dataset {imagenet_dataset}")
    else:
        imagenet_dataset = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data['train']['samples_per_gpu'],
        shuffle=True,
        num_workers=cfg.data['train']['workers_per_gpu'],
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.data['eval']['samples_per_gpu'],
        shuffle=False,
        num_workers=cfg.data['eval']['workers_per_gpu'],
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data['test']['samples_per_gpu'],
        shuffle=False,
        num_workers=cfg.data['test']['workers_per_gpu'],
        pin_memory=True
    )
    if imagenet_dataset is not None:
        imagenet_dataloader = DataLoader(
            imagenet_dataset,
            batch_size=cfg.data["imagenet"]["samples_per_gpu"],
            shuffle=False,
            num_workers=cfg.data["imagenet"]["workers_per_gpu"],
            pin_memory=True
        )
    else:
        imagenet_dataloader = None

    data = dict(train=train_dataloader, eval=eval_dataloader, test=test_dataloader)
    if imagenet_dataloader is not None:
        data["imagenet"] = imagenet_dataloader

    # Load Model
    logger.info("Building model...")
    model = MODELS.build(cfg.model)
    model = model.cuda()
    logger.info(f"Built model: \n{model}")

    # Load Solver
    logger.info("Building solver...")
    solver = SOLVERS.build(model, cfg.solver, logger=logger)
    logger.info(f"Built solver: {solver}")

    solver.solve(data)
