# 快速训练

本教程将以`HMDB51`数据集为例，提供一个通过essmc2软件包和预训练模型进行快速finetunue的基本教程。

### 准备数据集

HMDB51是一个视频动作分类数据集，总共6849个视频，包含51类动作。具体可以参考[官网](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

#### 1. 下载视频和标注文件
运行前请确保当前目录为`EssentialMC2`
```shell
bash tools/data/hmdb51/download_hmdb51.sh
```

#### 2. 检查目录结构
在完成HMDB51数据集下载之后，HMDB51的文件结构如下
```
EssentialMC2
├── configs
├── essmc2
├── data
|   ├── hmdb51
|   |   ├── annotations
|   |   |   ├── brush_hair_test_split1.txt

|   |   ├── videos
|   |   |   ├── brush_hair
|   |   |   |   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi

│   │   |   ├── wave
│   │   │   |   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
```

### 准备配置文件

#### 1. 构建模型
使用预定义的视频分类模型`essmc2.models.networks.classifier.VideoClassifier` ，模型被分为3个部件：

- backbone：主干网络，通常用于进行视频输入的特征提取
- neck：用于将主干网络提取的特征转变为适合下游任务的部件
- head：用于具体任务的头部部件，这里使用分类头

以`configs/models/tada_r50_ssv2.py`为例
```python
model = dict(
    type="VideoClassifier",
    backbone=dict(type="ResNet3D_TAda"),
    neck=dict(type="GlobalAveragePooling", dim=3),
    head=dict(type="VideoClassifierHead", dim=2048, num_classes=174, dropout_rate=0.5)
)
```
该配置定义了一个视频分类器，其主干网络为`ResNet3D_Tada`，默认层数为50，分类头需要分类174类。

在我们的示例中，`HMDB51`数据集一共有51类，因此需要调整`num_classes`为51.

#### 2. 构建数据集
##### 2.1 构建数据流水线
数据流水线(pipeline)定义了如何读取和解码相关数据`io`，如何对视频和图像等数据进行必要的`augmentation`，以及最终以什么样的格式输入至网络

以下代码构建了一个基础的视频处理pipeline，其中
- DecodeVideoToTensor：读取本地/远程视频文件，解码并生成对应Tensor
- TensorToGPU：将Tensor转存至GPU，方面后续transform在GPU上进行推导，起到加速的作用
- VideoToTensor：将`uint8`类型的tensor转换为`float32`类型
- RandomResizedCropVideo：对视频数据进行随机resize和crop操作
- RandomHorizontalFlipVideo：对视频数据进行随机flip操作
- NormalizeVideo：将视频数据归一化
- Select：选择需要输入至网络的字段，`VideoClassifier`模型需要`video`和`gt_label`两类数据

```python
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
    dict(type='NormalizeVideo', mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    dict(type="Select", keys=["video", "gt_label"])
]
```

##### 2.2 构建数据集

这里需要注意的是`temporal_crops`和`spatial_crops`两个参数，具体请参考`Hmdb51`类注释。

```python
train_pipeline = ...
train = dict(
    type="Hmdb51",
    data_root_dir="data/hmdb51/videos",
    annotation_dir="data/hmdb51/annotations",
    temporal_crops=1,
    spatial_crops=1,
    mode="train",
    pipeline=train_pipeline
)
```

##### 2.3 构建dataloader
dataloader中需要注意`samples_per_gpu`和`workers_per_gpu`两个参数，前者表示每块GPU卡上的batch数量，后者表示启动多少个子进程进行数据读取的加速。

另外对于小数据集训练时，将`num_folds`参数设置为`N`（`N>1`)，可以允许一次训练epoch中将该数据集重复`N`次，并且允许在每次iter后调整学习率（需要配合`LrHook`钩子）
，是一个值的尝试的参数。

```python
train = ...
val = ...
data = dict(
    train=dict(
        samples_per_gpu=128,
        workers_per_gpu=8,
        dataset=train,
        num_folds=1
    ),
    eval=dict(
        samples_per_gpu=128,
        workers_per_gpu=8,
        dataset=val
    ),
    pin_memory=False
)
```


#### 3. 构建求解器
##### 3.1 设置优化器
使用pytorch内置的`SGD`优化器。
```python
optimizer = dict(
    type="SGD",
    lr=0.001,
    weight_decay=0.001
)
```

##### 3.2 设置lr调节器
使用pytorch内置的`CosineAnnealingLR`
```python
max_epoch = 100
lr_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=max_epoch,
    eta_min=0
)
```

##### 3.3 设置钩子`Hook`
`Hook`钩子是训练流程中重要的一个组件，可以在不侵入`Solver`实现的同时，实现多种需求
- BackwardHook：反向传播
- CheckpointHook：加载和保存训练状态
- LogHook：写日志
- TensorboardHook：写训练状态数据至tensorboard中
- LrHook：调整学习率
- DistSamplerHook：分布式训练时，用于进行数据重打散

`Hook`一般包括`before_solve`,`before_epoch`,`before_iter`,`after_iter`,`after_epoch`,`after_solve`这六个阶段，
配合solver当前状态可以准确在任意想插入钩子的地方进行相关代码调用。注意`Hook`一般是有优先级的，具体可以查阅`essmc2.hooks.__init__.py`文件。

```python
hooks = [
    dict(
        type='BackwardHook'
    ),
    dict(
        type='LogHook',
        log_interval=10
    ),
    dict(
        type='LrHook',
    ),
    dict(
        type='CheckpointHook',
        interval=1,
    ),
    dict(
        type='DistSamplerHook'
    )
]
```

##### 3.4 创建求解器`Solver`
求解器用于组织具体训练流程的调度方式，
- 可以是简单如`TrainValSolver`
- 也可以复杂如`papers/ICCV2021-NGC/impls/solvers.py`中的`NGCSolver`，该求解器在求解过程中会阶段性进行graph构图操作，用于识别噪声数据。

```python
optimizer = ...
lr_scheduler = ...
max_epochs = 100
num_folds = 1
hooks = ...
solver = dict(
    type='TrainValSolver',
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    max_epochs=max_epochs,
    num_folds=num_folds,
    hooks=hooks,
)
```

至此，全部需要的必备组件均构建完成，具体可以查看`configs/tutorials/finetune_hmdb51.py`

### 进行训练
#### 单机单卡训练
```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
	--config configs/tutorials/finetune_hmdb51.py
```

#### 单机多卡训练
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
	tools/train.py \
	--config configs/tutorials/finetune_hmdb51.py \
	--dist_launcher pytorch
```

#### 分布式训练
```shell
# srun, to be implemented
```