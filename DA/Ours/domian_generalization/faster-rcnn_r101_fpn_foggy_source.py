_base_ = [
    '../../_base_/models/faster-rcnn_r101_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/foggy_cityscapes.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8
