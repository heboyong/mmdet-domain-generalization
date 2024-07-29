# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')

backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

branch_field = ['sup_weak', 'sup_strong', 'unsup_weak', 'unsup_strong']

sup_weak_pipeline = [
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

sup_strong_pipeline = [
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='AlbuDomainAdaption', domain_adaption_type='ALL',
         target_dir='data/cityscapes/JPEGImages', p=0.5),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

unsup_weak_pipeline = [
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

unsup_strong_pipeline = [
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]

sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomCrop', crop_type='absolute', crop_size=(1024, 1024),
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup_weak=sup_weak_pipeline,
        sup_strong=sup_strong_pipeline,
    )
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(type='RandomResize', scale=[(1024, 800), (1024, 1024)], keep_ratio=True),
    dict(type='RandomCrop', crop_type='absolute', crop_size=(1024, 1024),
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_weak=unsup_weak_pipeline,
        unsup_strong=unsup_strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

batch_size = 2
num_workers = 8

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file='cityscapes/train.json',
    data_prefix=dict(img='cityscapes/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=sup_pipeline)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file='diffusion_aug/cityscapes/train_aug.json',
    data_prefix=dict(img='diffusion_aug/cityscapes/aug/'),
    pipeline=unsup_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 1]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='cityscapes/test.json',
        data_prefix=dict(img='cityscapes/JPEGImages/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'cityscapes/test.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
