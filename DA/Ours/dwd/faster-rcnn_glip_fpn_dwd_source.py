_base_ = [
    '../../_base_/models/faster-rcnn_glip_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/dwd.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8

optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(val_interval=10000)