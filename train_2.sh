#!/bin/sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29600 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/sim10k/faster-rcnn_r101+dift_fpn_sim10k_semi_base_e2e_aug.py


