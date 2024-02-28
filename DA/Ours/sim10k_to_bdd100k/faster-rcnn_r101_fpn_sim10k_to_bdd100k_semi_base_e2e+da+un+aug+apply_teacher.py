_base_ = [
    '../../_base_/models/semi_faster_rcnn_r101_fpn.py',
    '../../_base_/da_setting/semi_e2e_20k_0.1backbone.py',
    '../../_base_/datasets/sim10k_to_bdd100k/semi_sim10k_to_bdd100k.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 1

model = dict(
    _delete_=True,
    type='DomainAdaptationDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBase', burn_up_iters=_base_.burn_up_iters),  # []
        # 训练模式选择，从0开始
        da_cfg=dict(
            use_uda=True,
            use_uncertainty=True,
            da_start_iters=_base_.da_start_iters,
            input_mode=[
                'backbone[-1]',  # backbone最后一层
                'neck[-1]',  # neck最后一层
                'neck_all',  # 所有neck共用一个域分类器
                'neck_cat',  # neck concat维度
            ],
            input_select=1,
            daloss_weight=1.0,
        ),
        domain_aug_cfg=dict(
            apply_domain_aug=True,
            apply_teacher_aug=True,
            adaptive_threshold=False,
            domain_aug_methods=['FDA', 'HM', 'PDA']
        )
    )
)
