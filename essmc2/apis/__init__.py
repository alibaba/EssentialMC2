"""
A standard config file should include
* data: shows how to build data
    - train: show how to build train dataloader, None for EvaluationSolver
        - samples_per_gpu: int, default is 2
        - workers_per_gpu: int, default is 2
        - dataset: show how to build dataset
            - type: str
            - mode: str, 'train'
            - other kwargs
    - eval: show how to build eval dataloader, optional
        - samples_per_gpu: int, default is 2
        - workers_per_gpu: int, default is 2
        - dataset: show how to build dataset
            - type: str
            - mode: str, 'eval'
            - other kwargs
    - pin_memory: bool, default is False
* model: shows how to build a model
* solver: shows how to train a model
    - type: TrainValSolver/EvaluationSolver
    For EvaluationSolver:
        - eval_interval: int, default is 1
        - do_final_eval: bool, default is False
        - save_eval_data: bool, default is False
        - eval_metric_cfg: dict / List[dict], optional
            A dict contains
                * metric: dict(type='accuracy', **kwargs), describes how a metric function is built
                * keys: Sequence[str], describes the data key names the metric function requires
        - eval_keys: Sequence[str], optional
    - optimizer: dict, optional, describes how to build an optimizer
    - lr_scheduler: dict, optional, describes how to build a learning rate scheduler
    - resume_from: str, optional, describes if training resume from a checkpoint
    - work_dir: str, optional, describes where to save
    - envs: dict, optional, some hyper parameters can be set here
    - max_epochs: int, default is 1
    - num_folds: int, default is 1, number of training dataset fold numbers, affect lr / log / evaluation, etc...
    - hooks: Dict[dict], List[dict], optional
        Recommend Dict[dict]
            * backward
            * lr
            * log
            * ckpt
            * tensorboard
            * sampler
* seed: for randomness and reproducibility, optional
* dist：includes distribute information, optional
    - distributed: bool, default is False
    - sync_bn: bool, default is False
    - launcher: "pytorch", "pai", default is None
    - backend: "nccl", default is None
* file_systems: Dict[str, fs_config], List[fs_config], fs_config, optional, default is None
* cudnn: optional
    deterministic: bool, default is False
    benchmark: bool, default is False
"""