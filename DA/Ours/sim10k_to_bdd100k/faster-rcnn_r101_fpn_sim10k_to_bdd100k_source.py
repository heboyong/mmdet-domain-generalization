_base_ = [
    '../../_base_/models/faster-rcnn_r101_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/sim10k_to_bdd100k/sim10k_to_bdd100k_source.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 1
