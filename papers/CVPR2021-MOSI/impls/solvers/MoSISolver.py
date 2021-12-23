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
        bn_params_names = []
        non_bn_params = []
        non_bn_params_names = []
        head_params = []
        head_params_names = []
        no_weight_decay_params = []
        no_weight_decay_params_names = []
        for name, param in self.model.named_parameters(recurse=True):
            if "bn" in name or "norm" in name:
                bn_params.append(param)
                bn_params_names.append(name)
            elif "embd" in name or "cls_token" in name:
                no_weight_decay_params.append(param)
                no_weight_decay_params_names.append(name)
            elif "head" in name:
                head_params.append(param)
                head_params_names.append(name)
            else:
                non_bn_params.append(param)
                non_bn_params_names.append(name)

        optim_params = [
            {"params": bn_params, "weight_decay": self.bn_weight_decay, "names": bn_params_names},
            {"params": non_bn_params, "names": non_bn_params_names},
            {"params": head_params, "names": head_params_names},
            {"params": no_weight_decay_params, "names": no_weight_decay_params_names, "weight_decay": 0.0}

        ]

        return optim_params
