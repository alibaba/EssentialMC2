_delete_ = ['MEAN', 'STD', 'train_dataset', 'test_dataset']

MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)

train_dataset = dict(
    type='CifarNoisyDataset',
    mode='train',
    root_dir='./datasets/cifar-100-python',
    cifar_type='cifar100',
    noise_mode='sym',
    noise_ratio=0.2,
    pipeline=[
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
)

test_dataset = dict(
    type='CifarNoisyDataset',
    mode='test',
    root_dir='./datasets/cifar-100-python',
    cifar_type='cifar100',
    pipeline=[
        dict(type='ImageToTensor'),
        dict(type='Normalize', mean=MEAN, std=STD),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Select', keys=['img', 'gt_label'])
    ]
)

data = dict(
    train=dict(
        samples_per_gpu=512,
        workers_per_gpu=8,
        dataset=train_dataset
    ),
    test=dict(
        samples_per_gpu=512 * 4,
        workers_per_gpu=8,
        dataset=test_dataset
    ),
    eval=dict(
        samples_per_gpu=512 * 4,
        workers_per_gpu=8,
    )
)
