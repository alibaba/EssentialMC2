# 快速使用

1. 安装软件包
```shell
pip install essmc2
```

2. 快速进行视频分类任务 

预训练模型可以从这里下载 [MODEL_ZOO](https://github.com/alibaba/EssentialMC2/blob/main/MODEL_ZOO.md#finetuned)
```python
# Import all the required components
from essmc2.models import MODELS
from essmc2.transforms import TRANSFORMS
from essmc2.utils.video_reader import EasyVideoReader

# Load pretrained model
# r2d3ds_pt_hmdb_ft_hmdb_4693_public can be found in MODEL_ZOO
model = MODELS.build(dict(
    type="VideoClassifier",
    backbone=dict(type='ResNetR2D3D'),
    neck=dict(type="GlobalAveragePooling", dim=3),
    head=dict(type="VideoClassifierHead", dim=256, num_classes=51),
    pretrain='r2d3ds_pt_hmdb_ft_hmdb_4693_public.pyth'
))
model.eval()
model = model.cuda()

# Load transforms
pipeline = TRANSFORMS.build([
    dict(type='TensorToGPU', keys=('video', )),
    dict(type='VideoToTensor'),
    dict(type='ResizeVideo', size=128),
    dict(type='CenterCropVideo', size=112),
    dict(type='NormalizeVideo', mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
])

# The video is selected from HMDB51 dataset.
vr = EasyVideoReader("April_09_brush_hair_u_nm_np1_ba_goo_0.avi", 16, '64/30', transforms=pipeline)

for data in vr:
    logits = model(data['video'].unsqueeze(0))
    print("{:.2f}-{:.2f}, {}".format(data['meta']['start_sec'], data['meta']['end_sec'], logits.topk(k=5).indices[0]))

    
# Result
# 0.00-2.13, tensor([ 0, 22,  4,  3, 25], device='cuda:0')
# 2.13-4.27, tensor([ 0, 22,  4,  3, 25], device='cuda:0')
# 4.27-6.40, tensor([ 0, 22,  4,  3, 25], device='cuda:0')
# 6.40-8.53, tensor([ 0, 22,  4,  3, 25], device='cuda:0')
# 8.53-10.67, tensor([ 0, 22,  4,  3, 25], device='cuda:0')
# 10.67-12.80, tensor([ 0, 22,  4,  3, 25], device='cuda:0')

```