_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/sim10k/sim10k.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 1
