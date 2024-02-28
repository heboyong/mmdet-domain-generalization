#!/bin/sh

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29700 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/synscapes_to_city/faster-rcnn_r101_fpn_synscapes_to_city_source.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29700 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/synscapes_to_city/faster-rcnn_dift_fpn_synscapes_to_city_source.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29700 ./tools/train.py --launcher pytorch ${@:3} \
DA/Ours/synscapes_to_city/faster-rcnn_r101_fpn_synscapes_to_city_semi_base_e2e.py
