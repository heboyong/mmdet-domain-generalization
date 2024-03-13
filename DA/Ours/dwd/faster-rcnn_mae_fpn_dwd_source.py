_base_ = [
    '../../_base_/models/faster-rcnn_mae_fpn.py',
    '../../_base_/da_setting/da_20k_0.1backbone.py',
    '../../_base_/datasets/domain_generalization/dwd.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth'

train_cfg = dict(val_interval=10000)