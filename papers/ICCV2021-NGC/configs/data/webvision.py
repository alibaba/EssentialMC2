_delete_ = ['MEAN', 'STD', 'train_pipeline', 'test_pipeline', 'train_dataset', 'test_dataset', 'imagenet_test_dataset']

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

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
train_dataset = dict(
    type='Webvision',
    mode='train',
    root_dir='./datasets/webvision_mini',
    num_classes=50,
    pipeline=train_pipeline
)
test_dataset = dict(
    type='Webvision',
    mode='test',
    root_dir='./datasets/webvision_mini',
    num_classes=50,
    pipeline=test_pipeline
)
imagenet_test_dataset = dict(
    type="ImageNet",
    mode="val",
    root_dir='./datasets/webvision_mini/imagenet',
    num_classes=50,
    pipeline=test_pipeline
)
# Define dataloader
data = dict(
    train=dict(
        samples_per_gpu=256,
        workers_per_gpu=8,
        dataset=train_dataset
    ),
    test=dict(
        samples_per_gpu=256 * 4,
        workers_per_gpu=8,
        dataset=test_dataset
    ),
    eval=dict(
        samples_per_gpu=256 * 4,
        workers_per_gpu=8,
    ),
    imagenet=dict(
        samples_per_gpu=256 * 4,
        workers_per_gpu=8,
        dataset=imagenet_test_dataset
    )
)
