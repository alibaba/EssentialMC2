model = dict(
    type="VideoClassifier",
    backbone=dict(
        type='ResNetR2D3D',
        depth=18,
        bn_params=dict(eps=1e-3, momentum=0.1),
    ),
    neck=dict(type="GlobalAveragePooling", dim=3),
    head=dict(type="VideoClassifierHead", dim=256, num_classes=51, dropout_rate=0.5),
    topk=(1, 5)
)
