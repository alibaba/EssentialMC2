from get_data_definition import get_data_definition_finetune
from get_model_definition import get_classifier_definition
from get_solver_definition import get_solver_definition

hyper_params = dict(
    seed=123,
    samples_per_gpu=48,
    workers_per_gpu=4,
    max_epochs=300,
    num_folds=30,
    num_classes=51,
    db_name="Hmdb51",
    branch_name="R2Plus1DBranch",
    data_root_dir="datasets/hmdb51/videos",
    annotation_dir="datasets/hmdb51/anno_lists",
    lr=0.00075,
    warmup_start_lr=0.000075,
    weight_decay=0.001,
    bn_weight_decay=0
)

data = get_data_definition_finetune(hyper_params)

model = get_classifier_definition(hyper_params)

solver = get_solver_definition(hyper_params)

seed = hyper_params['seed']
