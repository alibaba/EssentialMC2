from _updater import update_data, update_model, update_solver

_base_ = ['./data/cifar100_noise.py',
          './model/preresnet.py',
          './solver/default_solver.py']
_delete_ = 'hyper_params'

hyper_params = dict(
    seed=123,
    # Data
    dataset_root='./datasets/cifar-100-python',
    dataset_name='cifar100',
    noise_mode='sym',
    noise_ratio=0.9,
    openset=False,
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
    high_threshold=0.6,
    do_aug=True,
)

data = update_data(hyper_params)

model = update_model(hyper_params)

solver = update_solver(hyper_params)

seed = hyper_params['seed']
