MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)


def get_data_definition(hyper_params: dict):
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
            label_mode=hyper_params["label_mode"],
            decouple=True,
            zero_out=hyper_params["zero_out"],
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
            num_speeds=hyper_params["num_speeds"],
            mode='test',
            aspect_ratio=(1, 1),
            distance_jitter=(1, 1),
            data_mode='xy',
            label_mode=hyper_params["label_mode"],
            decouple=True,
            zero_out=hyper_params["zero_out"],
            static_mask_enable=True,
            mask_size_ratio=(0.3, 0.5),
            frame_size_standardize_enable=True,
            standard_size=320,
            transforms=mosi_test_pipeline,
        ),
        dict(type="Select", keys=["video", "mosi_label"])
    ]
    train = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="train",
        pipeline=train_pipeline
    )
    val = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="val",
        pipeline=train_pipeline
    )
    test = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="test",
        pipeline=test_pipeline
    )

    data = dict(
        train=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=train,
            num_folds=hyper_params["num_folds"]
        ),
        test=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=test
        ),
        eval=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=val
        ),
        pin_memory=False
    )

    return data


def get_data_definition_finetune(hyper_params: dict):
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
        dict(type="Select", keys=["video", "gt_label"])
    ]
    train = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="train",
        pipeline=train_pipeline
    )
    val = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="val",
        pipeline=test_pipeline
    )
    test = dict(
        type=hyper_params["db_name"],
        data_root_dir=hyper_params["data_root_dir"],
        annotation_dir=hyper_params["annotation_dir"],
        temporal_crops=1,
        spatial_crops=1,
        mode="test",
        pipeline=test_pipeline
    )

    data = dict(
        train=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=train,
            num_folds=hyper_params["num_folds"]
        ),
        test=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=test
        ),
        eval=dict(
            samples_per_gpu=hyper_params["samples_per_gpu"],
            workers_per_gpu=hyper_params["workers_per_gpu"],
            dataset=val
        ),
        pin_memory=False
    )

    return data
