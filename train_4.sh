#!/bin/sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/dwd/faster-rcnn_swin+dift_fpn_dwd_semi_base_e2e_aug.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/dwd/faster-rcnn_vit+dift_fpn_dwd_semi_base_e2e_aug.py

