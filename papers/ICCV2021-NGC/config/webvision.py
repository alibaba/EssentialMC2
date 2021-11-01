MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

hyper_params = dict(
    seed=123,
    # Data
    dataset_root='./datasets/webvision_mini',
    dataset_name='webvision',
    openset=False,
    batch_size=256,
    workers_per_gpu=8,
    imagenet_root='./datasets/webvision_mini/imagenet',
    # Model
    num_classes=50,
    feature_dim=64,
    alpha=0.5,
    data_parallel=True,
    # Training
    max_epochs=80,
    lr=0.2,
    # Solver
    temperature=0.3,
    warmup_epoch=15,
    knn_neighbors=100,
    low_threshold=0.02,
    high_threshold=0.8,
    do_aug=False,
)

# Define data pipeline
train_pipeline = [
    dict(type='LoadPILImageFromFile'),
    dict(type='RandomResizedCrop', size=299, scale=(0.4, 1.0), output_key='img_aug'),
    dict(type='RandomHorizontalFlip', input_key='img_aug', output_key='img_aug'),
    dict(type='AugMix',
         mean=MEAN, std=STD,
         input_key='img_aug', output_key='img_aug'),
    dict(type='RandomResizedCrop', size=299, scale=(0.4, 1.0)),
    dict(type='RandomHorizontalFlip'),
    dict(type='ImageToTensor'),
    dict(type='Normalize', mean=MEAN, std=STD),
    dict(type='ToTensor', keys=['gt_label', 'index']),
    dict(type='Select', keys=['img', 'img_aug', 'gt_label', 'index'])
]
test_pipeline = [
    dict(type='LoadPILImageFromFile'),
    dict(type='Resize', size=320),
    dict(type='CenterCrop', size=299),
    dict(type='ImageToTensor'),
    dict(type='Normalize', mean=MEAN, std=STD),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Select', keys=['img', 'gt_label'])
]
# Define dataset
train = dict(
    type='Webvision',
    mode='train',
    root_dir=hyper_params['dataset_root'],
    num_classes=hyper_params['num_classes'],
    pipeline=train_pipeline
)
test = dict(
    type='Webvision',
    mode='test',
    root_dir=hyper_params['dataset_root'],
    num_classes=hyper_params['num_classes'],
    pipeline=test_pipeline
)
imagenet_test = dict(
    type="ImageNet",
    mode="val",
    root_dir=hyper_params['imagenet_root'],
    num_classes=hyper_params['num_classes'],
    pipeline=test_pipeline
)
# Define dataloader
data = dict(
    train=dict(
        samples_per_gpu=hyper_params['batch_size'],
        workers_per_gpu=hyper_params['workers_per_gpu'],
        dataset=train
    ),
    test=dict(
        samples_per_gpu=hyper_params['batch_size'] * 4,
        workers_per_gpu=hyper_params['workers_per_gpu'],
        dataset=test
    ),
    eval=dict(
        samples_per_gpu=hyper_params['batch_size'] * 4,
        workers_per_gpu=hyper_params['workers_per_gpu'],
    ),
    imagenet=dict(
        samples_per_gpu=hyper_params['batch_size'] * 4,
        workers_per_gpu=8,
        dataset=imagenet_test
    )
)
# Define model
model = dict(
    type="NGCNetwork",
    backbone=dict(
        type="InceptionResNetV2",
        use_pretrain=False
    ),
    neck=dict(
        type="GlobalAveragePooling"
    ),
    head=dict(
        type="NoisyContrastHead",
        in_channels=1536,
        num_classes=hyper_params['num_classes'],
        out_feat_dim=hyper_params['feature_dim']
    ),
    num_classes=hyper_params['num_classes'],
    alpha=hyper_params['alpha'],
    data_parallel=hyper_params['data_parallel']
)
# Define optimizer
optimizer = dict(
    type='SGD',
    lr=hyper_params['lr'],
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False
)
# Define lr scheduler
lr_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=hyper_params['max_epochs'],
    eta_min=0.0002
)
# Define hooks
hooks = [
    dict(
        type='BackwardHook'
    ),
    dict(
        type='LogHook',
        log_interval=20
    ),
    dict(
        type='LrHook',
    ),
    dict(
        type='CheckpointHook',
        interval=1,
    ),
]
# Define solver
solver = dict(
    type='NGCSolver',
    hyper_params=hyper_params,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    max_epochs=hyper_params['max_epochs'],
    hooks=hooks,
)

seed = hyper_params['seed']
