_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/cityscapes-c.py'
]
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
