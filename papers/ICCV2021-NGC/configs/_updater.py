def update_data(hyper_params):
    return dict(
        train=dict(
            samples_per_gpu=hyper_params['batch_size'],
            workers_per_gpu=hyper_params['workers_per_gpu'],
            dataset=dict(
                root_dir=hyper_params['dataset_root'],
                cifar_type=hyper_params['dataset_name'],
                noise_mode=hyper_params['noise_mode'],
                noise_ratio=hyper_params['noise_ratio']
            )
        ),
        test=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
            dataset=dict(
                root_dir=hyper_params['dataset_root'],
                cifar_type=hyper_params['dataset_name']
            )
        ),
        eval=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
        )
    )


def update_openset_data(hyper_params):
    return dict(
        train=dict(
            samples_per_gpu=hyper_params['batch_size'],
            workers_per_gpu=hyper_params['workers_per_gpu'],
            dataset=dict(
                root_dir=hyper_params['dataset_root'],
                cifar_type=hyper_params['dataset_name'],
                noise_mode=hyper_params['noise_mode'],
                noise_ratio=hyper_params['noise_ratio'],
                ood_noise_name=hyper_params['ood_noise_name'],
                ood_noise_root_dir=hyper_params['ood_noise_root_dir'],
                ood_noise_num=hyper_params['ood_noise_num_train']
            )
        ),
        test=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
            dataset=dict(
                root_dir=hyper_params['dataset_root'],
                cifar_type=hyper_params['dataset_name'],
                ood_noise_name=hyper_params['ood_noise_name'],
                ood_noise_root_dir=hyper_params['ood_noise_root_dir'],
                ood_noise_num=hyper_params['ood_noise_num_test']
            )
        ),
        eval=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
        )
    )


def update_webvision_data(hyper_params):
    return dict(
        train=dict(
            samples_per_gpu=hyper_params['batch_size'],
            workers_per_gpu=hyper_params['workers_per_gpu'],
        ),
        test=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
        ),
        eval=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=hyper_params['workers_per_gpu'],
        ),
        imagenet=dict(
            samples_per_gpu=hyper_params['batch_size'] * 4,
            workers_per_gpu=8,
        )
    )


def update_model(hyper_params):
    return dict(
        head=dict(num_classes=hyper_params['num_classes'], out_feat_dim=hyper_params['feature_dim']),
        num_classes=hyper_params['num_classes'],
        alpha=hyper_params['alpha'],
        data_parallel=hyper_params['data_parallel']
    )


def update_solver(hyper_params):
    return dict(
        hyper_params=hyper_params,
        optimizer=dict(lr=hyper_params['lr'], weight_decay=hyper_params.get('weight_decay') or 5e-4),
        lr_scheduler=dict(T_max=hyper_params['max_epochs']),
        max_epochs=hyper_params['max_epochs'],
    )
