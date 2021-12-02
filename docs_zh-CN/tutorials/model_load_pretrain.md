# 优雅地导入预训练模型

### 动机
使用`MODELS`注册器生成模型后，可能需要进行预训练参数的导入。使用pytorch
自带的`model.load(state_dict)`函数能满足大多数的需求。然而在以下几种场景下，
使用自带的导入函数将不满足需求：

1. 单机的模型希望导入分布式环境下训练的模型`DistributeParallel`，将因为`module`前缀导致导入失败;
2. 新的分类模型的`head`模块中调整了新的分类个数，将因为`fc`层参数大小不匹配导致导入失败;
3. 新的分类模型的`backbone`模块只希望导入自监督模型的`backbone`模块，一般情况下需要手动操作;
4. 读取存放于不同文件系统下的预训练参数，一般需要手动下载；

在这篇文章中，将介绍一种优雅地导入预训练模型的机制，来解决上述问题。

### 解决方法
在定义模型结构时，通过`pretrain`参数来指明如何导入预训练模型即可。

```python
import torch
from torch.nn.parallel import DataParallel
from essmc2.models import MODELS, BACKBONES

# Mock一个模型，并保存相关参数
model = MODELS.build(dict(
    type="Classifier",
    backbone=dict(type='ResNet'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(type='ClassifierHead', dim=2048, num_classes=100),
))

model = DataParallel(model)

torch.save({
    "state_dict": model.state_dict()
}, "pretrain.pth")

# 创建一个分类数不一样的新模型，并load参数
new_classifier = MODELS.build(dict(
    type="Classifier",
    backbone=dict(type="ResNet"),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(type='ClassifierHead', dim=2048, num_classes=200),
    pretrain="pretrain.pth"
))
# 2021-12-02 17:20:12,821 - essmc2 - INFO - Load pretrained model [Classifier] from pretrain.pth
# 2021-12-02 17:20:12,925 - essmc2 - WARNING - ignore keys from source:
# head.fc.weight: invalid shape, dst torch.Size([200, 2048]) vs. src torch.Size([100, 2048])
# head.fc.bias: invalid shape, dst torch.Size([200]) vs. src torch.Size([100])
# 2021-12-02 17:20:12,974 - essmc2 - WARNING - missing key in source state_dict: head.fc.weight, head.fc.bias

# 创建一个backbone，并load参数
new_backbone = BACKBONES.build(dict(
    type="ResNet",
    pretrain=dict(path='pretrain.pth', sub_level="backbone")
))
# 2021-12-02 17:21:58,616 - essmc2 - INFO - Load pretrained model [ResNet] from pretrain.pth

# 创建一个从url load参数的backbone
new_backbone_2 = BACKBONES.build(dict(
    type="ResNet",
    pretrain="https://download.pytorch.org/models/resnet50-0676ba61.pth"
))
# 2021-12-02 17:23:31,761 - essmc2 - INFO - Load pretrained model [ResNet] from https://download.pytorch.org/models/resnet50-0676ba61.pth
# Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to .cache/torch/hub/checkpoints/resnet50-0676ba61.pth
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:17<00:00, 5.97MB/s]
# 2021-12-02 17:23:53,279 - essmc2 - WARNING - unexpected key in source state_dict: fc.weight, fc.bias
```