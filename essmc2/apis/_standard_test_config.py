data = dict(
    eval=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        data=dict(
            type='',
            mode='eval',
        )
    ),
    pin_memory=False
)

model = dict(
    type='',
    pretrain=None
)

solver = dict(
    type='Evaluation',
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
        log=dict(type='LogHook', log_interval=10),
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
