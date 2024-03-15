_base_ = [
    '../../_base_/models/faster-rcnn_convnext-b_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/dwd.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(val_interval=4000)