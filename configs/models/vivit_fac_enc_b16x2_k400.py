model = dict(
    type="VideoClassifier",
    backbone=dict(type="ViVit_Fac_Enc", image_size=112, num_frames=16),
    neck=None,
    head=dict(type="TransformerHead", dim=768, num_classes=400, dropout_rate=0.0)
)
