_base_ = [
    '../../_base_/models/semi_faster_rcnn_mae+dift_fpn.py',
    '../../_base_/da_setting/semi_e2e_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/semi_dwd_aug.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 8
detector.dift_model.config = 'work_dirs_all/faster-rcnn_dift_fpn_dwd_source/faster-rcnn_dift_fpn_dwd_source.py'
detector.dift_model.pretrained_model = 'work_dirs_all/faster-rcnn_dift_fpn_dwd_source/iter_20000.pth'
detector.semi_train_cfg.student_pretrained = 'work_dirs_all/faster-rcnn_mae_fpn_dwd_source/iter_20000.pth'

model = dict(
    _delete_=True,
    type='DomainAdaptationDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDift', burn_up_iters=_base_.burn_up_iters),  # []
        # 训练模式选择，从0开始
        da_cfg=dict(
            use_uda=False,
            use_uncertainty=False,
            da_start_iters=_base_.da_start_iters,
            input_mode=[
                'backbone[-1]',  # backbone最后一层
                'neck[-1]',  # neck最后一层
                'neck_all',  # 所有neck共用一个域分类器
                'neck_cat',  # neck concat维度
            ],
            input_select=1,
            daloss_weight=1.0
        ),
        domain_aug_cfg=dict(
            apply_domain_aug=False,
            apply_teacher_aug=False,
            adaptive_threshold=False,
            domain_aug_methods=['FDA', 'HM', 'PDA']
        )
    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

train_cfg = dict(val_interval=4000)