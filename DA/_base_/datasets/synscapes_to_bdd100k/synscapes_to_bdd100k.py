# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck')

backend_args = None

branch_field = ['source', 'target']

source_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(512, 512),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        source=dict(type='PackDetInputs'))
]
target_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(512, 512),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        target=dict(type='PackDetInputs'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 8
num_workers = 2
source_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file='SynScapes/train.json',
    data_prefix=dict(img='SynScapes/img/rgb-2k/'),
    pipeline=source_pipeline)

target_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file='BDD100K/bdd100k/train.json',
    data_prefix=dict(img='BDD100K/bdd100k/bdd100k/bdd100k/images/100k/train/'),
    pipeline=target_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[2, 2]),
    dataset=dict(
        type='ConcatDataset', datasets=[source_dataset, target_dataset]))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='BDD100K/bdd100k/val.json',
        data_prefix=dict(img='BDD100K/bdd100k/bdd100k/bdd100k/images/100k/val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'BDD100K/bdd100k/val.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
