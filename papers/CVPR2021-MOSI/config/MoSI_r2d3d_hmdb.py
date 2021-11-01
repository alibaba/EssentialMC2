from get_data_definition import get_data_definition
from get_model_definition import get_mosinet_definition
from get_solver_definition import get_solver_definition

hyper_params = dict(
    seed=123,
    num_speeds=5,
    label_mode="joint",
    zero_out=False,
    samples_per_gpu=10,
    workers_per_gpu=4,
    max_epochs=100,
    num_folds=20,
    branch_name="R2D3DBranch",
    db_name="Hmdb51",
    data_root_dir="datasets/hmdb51/videos",
    annotation_dir="datasets/hmdb51/anno_lists",
    lr=0.001,
    warmup_start_lr=0.0001,
    weight_decay=1e-4,
    bn_weight_decay=0.0
)

hyper_params["num_classes"] = (hyper_params["num_speeds"] - 1) * 2 + 1 * (not hyper_params["zero_out"])

data = get_data_definition(hyper_params)

model = get_mosinet_definition(hyper_params)

solver = get_solver_definition(hyper_params)

seed = hyper_params['seed']
