custom_imports = dict(
    imports=['mmdet.models.plugins.custom_lineformer'],
    allow_failed_imports=False)
_base_ = [
    '../_base_/datasets/whu_line.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='InstanceLineSegmentor',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),
        style='caffe',),
    neck=dict(
        type='MSDeformAttnFPN',
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        init_cfg=None),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 128), (128, 256), (256, 1e6)),
        strides=[8, 16, 32],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    line_head=dict(
        type='LineFormerHead',
        in_channel=256,
        out_channel=4,
        num_query=30,
        roi_wh=(32, 32),
        expand_scale=1.1,
        #regress_ranges=((-1, 128), (128, 256), (256, 1e6)),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(32, 32), sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4]),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        line_predictor=dict(
            type='LinePredictor',
            refer_points_nums=36,
            pred_line_with_pos=True,
            update_pos_embed=True,
            line_decoder=dict(
                type='LineDecoder',
                return_intermediate=True,
                num_layers=3,
                hidden_dim=256,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                     'ffn', 'norm')),),
            order_decoder=dict(
                type='OrderDecoder',
                return_intermediate=False,
                num_layers=3,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                     'ffn', 'norm')),)),
        line_loss=dict(
            type='LineLoss',
            beta=0.01,
            loss_weight=5.0),
        score_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 0.1],
            loss_weight=1.0,
        ),
        reference_points_loss=dict(
            type='SmoothL1Loss',
            beta=0.01,
            loss_weight=1.0),
        order_loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0 / 36,
        ),
    ),
    detector_fpn_start_level=0, #start fron P3
    line_fpn=False,
    line_fpn_start_level=0,
    # training and testing settings
    train_cfg=dict(
        line_assigner=dict(
            type='LineAssigner',
            score_weight=1.,
            line_weight=5.
        ),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1344, 816), (1344, 1344)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='LineSampleWithAlignReference', point_nums=36, reset_bbox=True, with_reference_points=True),
    dict(type='LineDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_lines', 'reference_points', 'matched_idxs']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
find_unused_parameters = True
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=1.0, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[45])
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(metric=['bbox', 'segm'], interval=10)
checkpoint_config = dict(interval=5)
