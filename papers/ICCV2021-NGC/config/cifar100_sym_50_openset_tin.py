MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)

hyper_params = dict(
    seed=123,
    # Data
    dataset_root='./datasets/cifar-100-python',
    dataset_name='cifar100',
    noise_mode='sym',
    noise_ratio=0.5,
    openset=True,
    open_noise_name="tiny_imagenet",
    open_noise_root="./datasets/",
    open_noise_num_train=20000,
    open_noise_num_test=10000,
    batch_size=512,
    workers_per_gpu=8,
    # Model
    num_classes=100,
    feature_dim=64,
    alpha=8.0,
    data_parallel=False,
    # Training
    max_epochs=300,
    lr=0.15,
    # Solver
    temperature=0.3,
    warmup_epoch=30,
    knn_neighbors=200,
    low_threshold=0.01,
    high_threshold=0.8,
    do_aug=True,
)

# Define data pipeline
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4, output_key='img_aug'),
    dict(type='RandomHorizontalFlip', input_key='img_aug', output_key='img_aug'),
    dict(type='AugMix',
         mean=MEAN, std=STD,
         input_key='img_aug', output_key='img_aug'),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ImageToTensor'),
    dict(type='Normalize', mean=MEAN, std=STD),
    dict(type='ToTensor', keys=['gt_label', 'index']),
    dict(type='Select', keys=['img', 'img_aug', 'gt_label', 'index'])
]
test_pipeline = [
    dict(type='ImageToTensor'),
    dict(type='Normalize', mean=MEAN, std=STD),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Select', keys=['img', 'gt_label'])
]
# Define dataset
train = dict(
    type='CifarNoisyOpensetDataset',
    mode='train',
    root_dir=hyper_params['dataset_root'],
    cifar_type=hyper_params['dataset_name'],
    noise_mode=hyper_params['noise_mode'],
    noise_ratio=hyper_params['noise_ratio'],
    ood_noise_name=hyper_params['open_noise_name'],
    ood_noise_root_dir=hyper_params['open_noise_root'],
    ood_noise_num=hyper_params['open_noise_num_train'],
    pipeline=train_pipeline,
)
test = dict(
    type='CifarNoisyOpensetDataset',
    mode='test',
    root_dir=hyper_params['dataset_root'],
    cifar_type=hyper_params['dataset_name'],
    ood_noise_name=hyper_params['open_noise_name'],
    ood_noise_root_dir=hyper_params['open_noise_root'],
    ood_noise_num=hyper_params['open_noise_num_test'],
    pipeline=test_pipeline,
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
    )
)
# Define model
model = dict(
    type="NGCNetwork",
    backbone=dict(
        type="PreResNet",
        depth=18,
        use_pretrain=False
    ),
    neck=dict(
        type="GlobalAveragePooling"
    ),
    head=dict(
        type="NoisyContrastHead",
        in_channels=512,
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
    weight_decay=5e-4,
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
