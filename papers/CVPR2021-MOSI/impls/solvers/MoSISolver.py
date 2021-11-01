# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from essmc2.solvers import SOLVERS
from essmc2.solvers import TrainValSolver


@SOLVERS.register_class()
class MoSISolver(TrainValSolver):
    def __init__(self, model, bn_weight_decay=0, **kwargs):
        self.bn_weight_decay = bn_weight_decay
        super(MoSISolver, self).__init__(model, **kwargs)

    def get_optim_parameters(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.model.named_parameters(recurse=True):
            if "bn" in name or "norm" in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)

        optim_params = [
            {"params": non_bn_params},
            {"params": bn_params, "weight_decay": self.bn_weight_decay}
        ]

        return optim_params
