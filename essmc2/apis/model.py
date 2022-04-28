import logging
from typing import Optional

import torch.nn
from torch.nn.parallel import DistributedDataParallel

from essmc2.models import MODELS
from essmc2.utils.config import Config


def get_model(cfg: Config, logger: Optional[logging.Logger] = None):
    model = MODELS.build(cfg.model)
    if cfg.dist.distributed and cfg.dist.get("sync_bn") is True:
        logger.info("Convert BatchNorm to Synchronized BatchNorm...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()

    if cfg.dist.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch.cuda.current_device()],
                                        output_device=torch.cuda.current_device())
    return model
