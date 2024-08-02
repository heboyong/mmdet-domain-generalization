_base_ = [
    '../../_base_/models/semi_faster_rcnn_r101+dift_fpn.py',
    '../../_base_/da_setting/semi_e2e_20k_0.1backbone.py',
    '../../_base_/datasets/dwd/semi_dwd_albu.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 7
detector.dift_model.config = 'DA/Ours/dwd/faster-rcnn_dift_fpn_dwd_source.py'
detector.dift_model.pretrained_model = 'work_dirs_dift/faster-rcnn_dift_fpn_dwd_source/iter_20000.pth'
detector.semi_train_cfg.student_pretrained = 'work_dirs_all/dwd/faster-rcnn_r101_fpn_dwd_source/iter_20000.pth'

model = dict(
    _delete_=True,
    type='DomainAdaptationDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDift', burn_up_iters=_base_.burn_up_iters),
        feature_loss_cfg=dict(
            enable_feature_loss=True,
            feature_loss_type='domain_classifier',
            # ['domain_classifier','mutual_information_maximization','l1','mse','kl']
            feature_loss_weight=1.0
        ),
    )
)

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(val_interval=4000)
