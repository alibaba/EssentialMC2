# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

# MultiFoldDistributedSampler class is modified from
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class MultiFoldDistributedSampler(Sampler):
    """Modified from DistributedSampler, which performs multi fold training for
    accelerating distributed training with large batches.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_folds (optional): Number of folds, if 1, will act same as DistributeSampler
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices

    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.
    """

    def __init__(self, dataset, num_folds=1, num_replicas=None, rank=None, shuffle=True):
        """
            When num_folds = 1, MultiFoldDistributedSampler degenerates to DistributedSampler.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_folds = num_folds
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_folds * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        for fold_idx in range(self.num_folds):
            g = torch.Generator()
            g.manual_seed(self.epoch + fold_idx)
            if self.shuffle:
                indices += torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices += list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MultiFoldRandomSampler(Sampler):
    """ Modified from torch.utils.data.sampler.RandomSampler.
    Add num_folds parameters.

    Args:
        dataset (Dataset): dataset to sample from
        num_folds (int): repeats of the dataset,, default = 1.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, dataset, num_folds=1, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None):
        self.dataset = dataset
        self.num_folds = num_folds
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset) * self.num_folds
        return self._num_samples

    def __iter__(self):
        n = len(self.dataset)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64,
                                         generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            indices = []
            for _ in range(self.num_folds):
                indices += torch.randperm(n, generator=self.generator).tolist()
            yield from indices

    def __len__(self):
        return self.num_samples


class MultiFoldSequentialSampler(Sampler):
    """Modified from torch.utils.data.sampler.SequentialSampler.
    Add num_folds parameters.

    Args:
        dataset (Dataset): dataset to sample from
    """

    def __init__(self, dataset, num_folds=1):
        self.dataset = dataset
        self.num_folds = num_folds

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices * self.num_folds
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset) * self.num_folds


class EvalDistributedSampler(Sampler):
    """Modified from DistributedSampler.

    Notice!
    1. This sampler should only be used in test mode.
    2. This sampler will pad indices or not pad, according to `padding` flag.
     In no padding mode, the last rank device may get samples less than given batch_size.
     The last rank device may have less iteration number than other rank.
     By the way, __len__ function may return a fake number.
    """

    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, padding: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.padding = padding

        self.perfect_num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.perfect_total_size = self.perfect_num_samples * self.num_replicas

        if self.padding:
            self._len = self.perfect_num_samples
        else:
            self._len = min((self.rank + 1) * self.perfect_num_samples, len(self.dataset)) \
                        - self.rank * self.perfect_num_samples

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.padding and len(indices) < self.perfect_total_size:
            padding_size = self.perfect_total_size - len(indices)
            indices += indices[:padding_size]

        # subsample
        indices = indices[self.rank * self.perfect_num_samples: (self.rank + 1) * self.perfect_num_samples]

        return iter(indices)

    def __len__(self) -> int:
        return self._len

    def set_epoch(self, epoch: int) -> None:
        pass
