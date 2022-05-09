data = dict(
    train=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
    ),
    eval=None,
    pin_memory=False
)

model = dict(
    type='',
    pretrain=None,
)

solver = dict(
    type='TrainValSolver',
    eval_interval=1,
    do_final_eval=False,
    save_eval_data=False,
    eval_metric_cfg=None,
    extra_keys=None,
    optimizer=None,
    lr_scheduler=None,
    resume_from=None,
    work_dir='./work_dir',
    envs=None,
    max_epochs=1,
    num_folds=1,
    hooks=dict(
        backward=dict(type='BackwardHook'),
        lr=dict(type='LrHook'),
        log=dict(type='LogHook', log_interval=10),
        ckpt=dict(type='CheckpointHook', interval=10),
        tensorboard=dict(type='TensorboardLogHook'),
        sampler=dict(type='DistSamplerHook')
    )
)

seed = None

dist = dict(
    distributed=False,
    sync_bn=False,
    launcher=None,
    backend=None,
)

file_systems = None

cudnn = dict(
    deterministic=False,
    benchmark=False,
)
