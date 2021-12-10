_reserve_ = ['solver', 'seed']

# Define hooks
hooks = dict(
    backward=dict(
        type='BackwardHook'
    ),
    log=dict(
        type='LogHook',
        log_interval=10
    ),
    lr=dict(
        type='LrHook',
        set_by_epoch=False,
        warmup_func="linear",
        warmup_epochs=10,
        warmup_start_lr=0.0001
    ),
    ckpt=dict(
        type='CheckpointHook',
        interval=1,
    ),
    sampler=dict(
        type='DistSamplerHook'
    )
)
# Define solver
solver = dict(
    type='MoSISolver',
    bn_weight_decay=0.0,
    optimizer=dict(),
    lr_scheduler=dict(),
    max_epochs=100,
    num_folds=1,
    hooks=hooks,
)

seed = 123
