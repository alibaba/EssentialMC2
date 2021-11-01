def get_r2p1d_backbone():
    backbone = dict(
        type="ResNet3D",
        depth=10,
        num_input_channels=3,
        num_filters=(64, 64, 128, 256, 512),
        kernel_size=((3, 7, 7), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        downsampling=(True, False, True, True, True),
        downsampling_temporal=(False, False, True, True, True),
        expansion_ratio=2,
        stem_name="R2Plus1DStem",
        branch_name="R2Plus1DBranch",
        non_local=(False, False, False, False, False),
        non_local_cfg=dict(),
        bn_params=dict(eps=1e-3, momentum=0.1),
        init_cfg=dict(),
        visual_cfg=dict(visualize=False, visualize_output_dir="")
    )
    return backbone


def get_r2d3d_backbone():
    backbone = dict(
        type="ResNet3D",
        depth=18,
        num_input_channels=3,
        num_filters=(64, 64, 128, 256, 256),
        kernel_size=((1, 7, 7), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)),
        downsampling=(True, False, True, True, True),
        downsampling_temporal=(False, False, False, True, True),
        expansion_ratio=2,
        stem_name="DownSampleStem",
        branch_name="R2D3DBranch",
        non_local=(False, False, False, False, False),
        non_local_cfg=dict(),
        bn_params=dict(eps=1e-3, momentum=0.1),
        init_cfg=dict(),
        visual_cfg=dict(visualize=False, visualize_output_dir="")
    )
    return backbone


def get_mosinet_definition(hyper_params):
    model = dict(
        type="MoSINet",
        backbone=get_r2d3d_backbone() if hyper_params["branch_name"] == "R2D3DBranch" else get_r2p1d_backbone(),
        neck=dict(type="GlobalAveragePooling", dim=3),
        head=dict(type="MoSIHead", dim=256 if hyper_params["branch_name"] == "R2D3DBranch" else 512,
                  num_classes=hyper_params["num_classes"], dropout_rate=0.5),
        label_mode=hyper_params["label_mode"],
        freeze_bn=False,
        topk=(1, 5)
    )
    return model


def get_classifier_definition(hyper_params):
    model = dict(
        type="VideoClassifier",
        backbone=get_r2d3d_backbone() if hyper_params["branch_name"] == "R2D3DBranch" else get_r2p1d_backbone(),
        neck=dict(type="GlobalAveragePooling", dim=3),
        head=dict(type="VideoClassifierHead", dim=256 if hyper_params["branch_name"] == "R2D3DBranch" else 512,
                  num_classes=hyper_params["num_classes"], dropout_rate=0.5),
        freeze_bn=False,
        topk=(1, 5)
    )
    return model
