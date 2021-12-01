# 构造模型推荐方案

### 动机
为了配合EvaluationSolver和TrainValSolver以及LogHook等组件需求，
需要模型（仅作用于注册于MODELS下的类，不包含BACKBONES、NECKS等基本组件）在不同模式下的有不同的输出形态。

### 构造方法
#### 基本结构
一般需要继承`essmc2.models.networks.train_module.TrainModule`类，这个类需要重写`forward_train`和`forward_test`这两个方法。
通过`nn.Module`的`training`属性来判别具体运行哪个函数。以下会通过视频分类模型`VideoClassifier`为例说明基本的构造方法。

```python
import torch
import functools
from essmc2.models import TrainModule, BACKBONES, NECKS, HEADS, LOSSES
from essmc2.utils.metrics import METRICS


class VideoClassifier(TrainModule):
    def __init__(self):
        super().__init__()
        self.backbone = BACKBONES.build(...)
        self.neck = NECKS.build(...)
        self.head = HEADS.build(...)
        
        self.loss = LOSSES.build(dict(type='CrossEntropy'))
        self.metric = METRICS.build(dict(...))
        self.activate_fn = functools.partial(torch.nn.functional.softmax, dim=1)
    
    def foward(self, video, **kwargs):
        return self.forward_train(video, **kwargs) if self.training else self.forward_test(video, **kwargs)
    
    def forward_test(self, video, gt_label=None):
        pass
    
    def forward_train(self, video, gt_label=None):
        pass
    
```

#### 测试模型
测试模式下，以`VideoClassifier`为例
* 基础要求：输入`video`字段，输出分类结果向量`logits`，满足tracing的需求；如果输出多个向量值，强烈建议以dict的格式（torch>=1.6.0之后支持jit.trace）输出相应键值对，方便`EvaluationSolver`进行处理；
* 额外要求：同时输入`gt_label`字段，则输出一个`dict`对象，包含`logits`键值，以及一些可选的衡量指标metrics，比如`accuracy`等；

```python
from essmc2.models import TrainModule
from collections import OrderedDict


class VideoClassifier(TrainModule):
    ...
    def forward_test(self, video, gt_label=None):
        # 基础要求，可直接用于部署 或者 jit.trace
        logits = self.activate_fn(self.head(self.neck(self.backbone(video))))
        if gt_label is None:
            return logits
        # 额外要求
        ret = OrderedDict()
        ret["logits"] = logits  # 包含logits
        ret.update(self.metric(logits, gt_label))  # 包含一些可选的衡量指标，其中的scalar值可以作为每轮的日志结果打印
        return ret
```

#### 训练模型
训练模型下，以`VideoClassifier`为例
* 基础要求：一般要求输入用于计算loss的groundtruth，比如`gt_label`，输出时需要输出一个`dict`对象，包含至少两个key：`loss`和`batch_size`。 
除此之外，允许同步输出一些衡量指标metrics，比如`accuracy`等；
* 额外要求：为了能够在training模式下，只输入`video`后也能直接运行。

```python
from essmc2.models import TrainModule
from collections import OrderedDict


class VideoClassifier(TrainModule):
    ...
    def forward_train(self, video, gt_label=None):
        probs = self.head(self.neck(self.backbone(video)))
        # 基础要求
        if gt_label is not None:
            ret = OrderedDict()
            loss = self.loss(probs, gt_label)
            ret["loss"] = loss
            ret["batch_size"] = video.size(0)
            ret.update(self.metric(probs, gt_label))
            return ret
        
        # 额外要求
        return self.activate_fn(probs)
```
