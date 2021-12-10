_delete_ = ['optimizer', 'lr_scheduler', 'hooks']

optimizer = dict(
    type='SGD',
    lr=0.15,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False
)

lr_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=300,
    eta_min=0.0002
)

hooks = dict(
    backward=dict(type='BackwardHook'),
    log=dict(type='LogHook', log_interval=20),
    lr=dict(type='LrHook'),
    ckpt=dict(type='CheckpointHook', interval=1)
)

solver = dict(
    type='NGCSolver',
    hyper_params=dict(),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    max_epochs=300,
    hooks=hooks,
)

seed = 123
