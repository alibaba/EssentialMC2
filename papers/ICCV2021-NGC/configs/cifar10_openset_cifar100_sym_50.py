from _updater import update_openset_data, update_model, update_solver

_base_ = ['./data/cifar10_noise_openset.py',
          './model/preresnet.py',
          './solver/default_solver.py']
_delete_ = 'hyper_params'

hyper_params = dict(
    seed=123,
    # Data
    dataset_root='./datasets/cifar-10-batches-py',
    dataset_name='cifar10',
    noise_mode='sym',
    noise_ratio=0.5,
    openset=True,
    ood_noise_name="cifar100",
    ood_noise_root_dir="./datasets/cifar-100-python/",
    ood_noise_num_train=20000,
    ood_noise_num_test=10000,
    batch_size=512,
    workers_per_gpu=8,
    # Model
    num_classes=10,
    feature_dim=64,
    alpha=8.0,
    data_parallel=False,
    # Training
    max_epochs=300,
    lr=0.15,
    # Solver
    temperature=0.3,
    warmup_epoch=5,
    knn_neighbors=30,
    low_threshold=0.1,
    high_threshold=0.8,
    do_aug=True,
)

data = update_openset_data(hyper_params)

model = update_model(hyper_params)

solver = update_solver(hyper_params)

seed = hyper_params['seed']
