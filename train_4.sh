#!/bin/sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/veis_to_city/faster-rcnn_r101_fpn_veis_to_city_source.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/veis_to_city/faster-rcnn_dift_fpn_veis_to_city_source.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29800 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/veis_to_city/faster-rcnn_r101_fpn_veis_to_city_semi_base_e2e.py
