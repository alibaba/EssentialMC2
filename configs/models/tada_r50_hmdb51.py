model = dict(
    type="VideoClassifier",
    backbone=dict(type="ResNet3D_TAda"),
    neck=dict(type="GlobalAveragePooling", dim=3),
    head=dict(type="VideoClassifierHead", dim=2048, num_classes=51, dropout_rate=0.5)
)
