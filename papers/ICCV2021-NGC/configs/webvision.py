from _updater import update_webvision_data, update_model, update_solver

_base_ = ['./data/webvision.py',
          './model/inceptionresnetv2.py',
          './solver/default_solver.py']
_delete_ = 'hyper_params'

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
    weight_decay=1e-4,
    # Solver
    temperature=0.3,
    warmup_epoch=15,
    knn_neighbors=100,
    low_threshold=0.02,
    high_threshold=0.8,
    do_aug=False,
)

data = update_webvision_data(hyper_params)

model = update_model(hyper_params)

solver = update_solver(hyper_params)

seed = hyper_params['seed']
