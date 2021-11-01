# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

from .base_solver import BaseSolver
from .train_val_solver import TrainValSolver
from .registry import SOLVERS
from .evaluation_solver import EvaluationSolver

__all__ = ['SOLVERS', 'BaseSolver', 'TrainValSolver', 'EvaluationSolver']
