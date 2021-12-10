_base_ = ['./data/hmdb51.py', './model/r2d3d_r18.py', './solver/default_solver.py']

data = dict(
    train=dict(
        samples_per_gpu=128,
        dataset=dict(
            data_root_dir='datasets/hmdb51/videos',
            annotation_dir='datasets/hmdb51/annotations'
        )
    ),
    eval=dict(
        samples_per_gpu=128,
        dataset=dict(
            data_root_dir='datasets/hmdb51/videos',
            annotation_dir='datasets/hmdb51/annotations'
        )
    )
)

model = dict(
    head=dict(num_classes=51)
)

solver = dict(
    bn_weight_decay=0.0,
    optimizer=dict(type='Adam', lr=0.002, betas=(0.9, 0.999), weight_decay=1e-3),
    lr_scheduler=dict(type='CosineAnnealingLR', T_max=300, eta_min=0),
    max_epochs=300,
    num_folds=30
)
