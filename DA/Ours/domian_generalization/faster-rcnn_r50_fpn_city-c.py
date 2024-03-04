_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/da_setting/da_80k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/cityscapes-c.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8

