_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/visda2018_to_coco/visda2018_to_coco_source.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 12
