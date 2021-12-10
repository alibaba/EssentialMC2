_base_ = ['./data/hmdb51_mosi.py', './model/r2p1d_r10.py', './solver/default_solver.py']

data = dict(
    train=dict(
        samples_per_gpu=5,
        dataset=dict(
            data_root_dir='datasets/hmdb51/videos',
            annotation_dir='datasets/hmdb51/annotations'
        )
    ),
    eval=dict(
        samples_per_gpu=5,
        dataset=dict(
            data_root_dir='datasets/hmdb51/videos',
            annotation_dir='datasets/hmdb51/annotations'
        )
    )
)

model = dict(
    type='MoSINet',
    head=dict(type='MoSIHead', num_classes=9),
    label_mode='joint'
)

solver = dict(
    bn_weight_decay=0.0,
    optimizer=dict(type='Adam', lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4),
    lr_scheduler=dict(type='CosineAnnealingLR', T_max=100, eta_min=0),
    max_epochs=100,
    num_folds=20
)
