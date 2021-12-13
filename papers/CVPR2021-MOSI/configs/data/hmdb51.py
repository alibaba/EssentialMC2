_reserve_ = 'data'

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

train_pipeline = [
    dict(
        type="DecodeVideoToTensor",
        num_frames=16,
        target_fps=30,
        sample_mode='interval',
        sample_interval=4,
        sample_minus_interval=False
    ),
    dict(
        type="TensorToGPU",
        keys=["video"]
    ),
    dict(type='VideoToTensor'),
    dict(
        type='RandomResizedCropVideo',
        size=112,
        scale=(168 * 168 / 256 / 340, 224 * 224 / 256 / 340),
        ratio=(0.857142857142857, 1.1666666666666667)
    ),
    dict(type='RandomHorizontalFlipVideo'),
    dict(
        type='ColorJitterVideo',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.25,
        grayscale=0.3,
        consistent=True,
        shuffle=True,
        gray_first=True
    ),
    dict(type='NormalizeVideo', mean=MEAN, std=STD),
    dict(type="Select", keys=["video", "gt_label"])
]
test_pipeline = [
    dict(
        type="DecodeVideoToTensor",
        num_frames=16,
        target_fps=30,
        sample_mode='interval',
        sample_interval=4,
        sample_minus_interval=False
    ),
    dict(
        type="TensorToGPU",
        keys=["video"]
    ),

    dict(type='VideoToTensor'),
    dict(
        type='AutoResizedCropVideo',
        size=112,
        scale=(224 / 256, 224 / 256),
    ),
    dict(type='NormalizeVideo', mean=MEAN, std=STD),
    dict(type="Select", keys=["video", "gt_label"], meta_keys=['video_path'])
]
train = dict(
    type='Hmdb51',
    data_root_dir='',  # to be filled
    annotation_dir='',  # to be filled
    temporal_crops=1,
    spatial_crops=1,
    mode="train",
    pipeline=train_pipeline
)
eval = dict(
    type='Hmdb51',
    data_root_dir='',  # to be filled
    annotation_dir='',  # to be filled
    temporal_crops=1,
    spatial_crops=1,
    mode="eval",
    pipeline=test_pipeline
)

data = dict(
    train=dict(
        samples_per_gpu=16,
        workers_per_gpu=4,
        dataset=train,
    ),
    eval=dict(
        samples_per_gpu=16,
        workers_per_gpu=4,
        dataset=eval
    ),
    pin_memory=False
)
