def get_solver_definition(hyper_params):
    # Define optimizer
    optimizer = dict(
        type='Adam',
        lr=hyper_params["lr"],
        betas=(0.9, 0.999),
        weight_decay=hyper_params["weight_decay"],
    )
    # Define lr scheduler
    lr_scheduler = dict(
        type='CosineAnnealingLR',
        T_max=hyper_params["max_epochs"],
        eta_min=0
    )
    # Define hooks
    hooks = [
        dict(
            type='BackwardHook'
        ),
        dict(
            type='LogHook',
            log_interval=10
        ),
        dict(
            type='LrHook',
            set_by_epoch=False,
            warmup_func="linear",
            warmup_epochs=10,
            warmup_start_lr=hyper_params["warmup_start_lr"]
        ),
        dict(
            type='CheckpointHook',
            interval=1,
        ),
        dict(
            type='DistSamplerHook'
        )
    ]
    # Define solver
    solver = dict(
        type='MoSISolver',
        bn_weight_decay=hyper_params["bn_weight_decay"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epochs=hyper_params["max_epochs"],
        num_folds=hyper_params["num_folds"],
        hooks=hooks,
    )

    return solver
