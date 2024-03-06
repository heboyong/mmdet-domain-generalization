# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.datasets.transforms import DiffusionAug, PackDetInputs
from mmdet.registry import HOOKS
from mmcv.transforms import Compose


@HOOKS.register_module()
class DiffusionAugSwitchHook(Hook):

    def __init__(
            self,
            burn_up_iters=12000,
    ) -> None:
        self.burn_up_iters = burn_up_iters
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_iter(self, runner, batch_idx=None, data_batch=None) -> None:
        iter = runner.iter
        train_loader = runner.train_dataloader
        iter_to_be_switched = (iter + 1) >= self.burn_up_iters
        if iter_to_be_switched and not self._has_switched:
            runner.logger.info('begin diffusion aug')
            pipeline = [
                dict(type='DiffusionAug'),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction',
                               'homography_matrix'))
            ]
            train_loader.dataset.datasets[-1].pipeline.transforms[-1].branch_pipelines['unsup_domain'] = \
                Compose(pipeline)
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        else:
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True


