model = dict(
    type="VideoClassifier",
    backbone=dict(type="ViVit_Fac_Enc", image_size=320, num_frames=32, drop_path=0.2),
    neck=None,
    head=dict(type="TransformerHeadx2", dim=768, num_classes=(97, 300), dropout_rate=0.0)
)
