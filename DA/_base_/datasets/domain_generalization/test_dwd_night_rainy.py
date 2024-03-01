# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'truck')

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='DWD/Night_Rainy/all.json',
        data_prefix=dict(img='DWD/Night_Rainy/JPEGImages/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'DWD/Night_Rainy/all.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
