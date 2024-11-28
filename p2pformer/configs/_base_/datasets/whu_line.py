# dataset settings
dataset_type = 'WhuDataset'
data_root = 'data/whu/'
classes = ('building')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    dict(type='AlignSampleBoundary', point_nums=128, reset_bbox=True),
    dict(type='ContourDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_masks', 'gt_polys', 'key_points_masks', 'key_points']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'overlap_val_40.json',
        img_prefix=data_root + 'overlap_val_40/',
        pipeline=test_pipeline),
    #test=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'annotations/train_val.json',
    #    img_prefix=data_root + 'train_val/',
    #    pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'overlap_val_40.json',
        img_prefix=data_root + 'overlap_val_40/',
        pipeline=test_pipeline),
    )
evaluation = dict(metric=['bbox', 'segm'], interval=20)
