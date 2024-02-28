_base_ = [
    '../../_base_/models/faster-rcnn_r101_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/veis_to_bdd100k/veis_to_bdd100k_source.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 7

train_cfg = dict(val_interval=4000)
