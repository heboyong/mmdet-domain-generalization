#!/bin/sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/domian_generalization/faster-rcnn_r101_fpn_sim10k_source_domain.py

