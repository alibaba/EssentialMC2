_reserve_ = 'data'

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

mosi_train_pipeline = [
    dict(type='VideoToTensor'),
    dict(
        type='ColorJitterVideo',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.25,
        grayscale=0.3,
        consistent=False,
        shuffle=True,
        gray_first=True
    ),
    dict(type='NormalizeVideo', mean=MEAN, std=STD),
    dict(type='RandomHorizontalFlipVideo'),
]
mosi_test_pipeline = [
    dict(type='VideoToTensor'),
    dict(type='NormalizeVideo', mean=MEAN, std=STD),
]
train_pipeline = [
    dict(
        type="DecodeVideoToTensor",
        num_frames=1,
        target_fps=30,
        sample_mode='interval',
        sample_interval=4,
        sample_minus_interval=False
    ),
    dict(
        type="TensorToGPU",
        keys=["video"]
    ),
    dict(
        type="MoSIGenerator",
        crop_size=112,
        num_frames=16,
        num_speeds=5,
        mode='train',
        aspect_ratio=(1, 1),
        distance_jitter=(1, 1),
        data_mode='xy',
        label_mode='joint',
        decouple=True,
        zero_out=False,
        static_mask_enable=True,
        mask_size_ratio=(0.3, 0.5),
        frame_size_standardize_enable=True,
        standard_size=320,
        transforms=mosi_train_pipeline,
    ),
    dict(type="Select", keys=["video", "mosi_label"])
]
test_pipeline = [
    dict(
        type="DecodeVideoToTensor",
        num_frames=1,
        target_fps=30,
        sample_mode='interval',
        sample_interval=4,
        sample_minus_interval=False
    ),
    dict(
        type="TensorToGPU",
        keys=["video"]
    ),
    dict(
        type="MoSIGenerator",
        crop_size=112,
        num_frames=16,
        num_speeds=5,
        mode='test',
        aspect_ratio=(1, 1),
        distance_jitter=(1, 1),
        data_mode='xy',
        label_mode='joint',
        decouple=True,
        zero_out=False,
        static_mask_enable=True,
        mask_size_ratio=(0.3, 0.5),
        frame_size_standardize_enable=True,
        standard_size=320,
        transforms=mosi_test_pipeline,
    ),
    dict(type="Select", keys=["video", "mosi_label"])
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
    pipeline=train_pipeline
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
