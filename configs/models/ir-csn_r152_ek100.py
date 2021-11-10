model = dict(
    type="VideoClassifier",
    backbone=dict(type="ResNet3D_CSN"),
    neck=dict(type="GlobalAveragePooling", dim=3),
    head=dict(type="VideoClassifierHeadx2", dim=2048, num_classes=(97, 300), dropout_rate=0.5),
    freeze_bn=True
)
