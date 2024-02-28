_base_ = [
    '../../_base_/models/faster-rcnn_r101_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/synscapes_to_bdd100k/synscapes_to_bdd100k_target.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 6

train_cfg = dict(val_interval=4000)
