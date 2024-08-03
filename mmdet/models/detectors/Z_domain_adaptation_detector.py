# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Tuple
import torch
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .Z_domain_classifier import FCDiscriminator_img, grad_reverse, da_focal_loss
from .Z_mutual_information_maximization import MIEstimator, MIMaxLoss
from .base import BaseDetector
from ..losses import KDLoss


@MODELS.register_module()
class DomainAdaptationDetector(BaseDetector):
    """Base class for semi-supervised detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 train_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert train_cfg is not None, "train_cfg is must not None"
        assert train_cfg.detector_cfg.get('type',
                                          None) is not None, "train_cfg.detector_cfg must use type select one detector"
        assert train_cfg.detector_cfg.get('type') in ['UDA', 'SemiBase', 'SoftTeacher', 'SemiBaseDift'], \
            "da_cfg type must select in ['UDA','SemiBase','SoftTeacher', 'SemiBaseDift]"
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.model = MODELS.build(detector)
        self.detector_name = detector.get('type')
        self.train_cfg = train_cfg
        self.enable_feature_loss = self.train_cfg.feature_loss_cfg.get('enable_feature_loss')

        # self.pkd_loss = KDLoss()

        if self.enable_feature_loss:
            self.feature_loss_type = self.train_cfg.feature_loss_cfg.get('feature_loss_type')
            self.feature_loss_weight = self.train_cfg.feature_loss_cfg.get('feature_loss_weight')

            if self.feature_loss_type in ['l1', 'mse', 'kl']:
                self.feature_loss = KDLoss(loss_weight=self.feature_loss_weight, loss_types=self.feature_loss_type)

            if self.feature_loss_type in ['domain_classifier']:
                self.domain_classifier = FCDiscriminator_img(256)
                self.da_loss = da_focal_loss

            if self.feature_loss_type in ['mutual_information_maximization']:
                self.mutual_information_maximization = MIEstimator(256)
                self.feature_loss = MIMaxLoss(loss_weight=self.feature_loss_weight)

        self.burn_up_iters = self.train_cfg.detector_cfg.get('burn_up_iters', 0)
        self.local_iter = 0

    @property
    def with_rpn(self):
        if self.with_student:
            return hasattr(self.model.student, 'rpn_head')
        else:
            return hasattr(self.student, 'rpn_head')

    @property
    def with_student(self):
        return hasattr(self.model, 'student')

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            input_mode=['source', #只是用源域训练
                    'backbone[-1]', #
                     'neck[-1]',
                    'neck_all',
                    'neck_cat',
                    'source+target'],  # 训练模式选择，从0开始

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        if self.train_cfg.detector_cfg.get('type') in ['SemiBase', 'SoftTeacher']:
            losses.update(**self.model.loss_by_gt_instances(
                multi_batch_inputs['sup_strong'], multi_batch_data_samples['sup_strong']))
            losses.update(
                **self.model.loss_by_gt_instances_domain(multi_batch_inputs['sup_domain'],
                                                         multi_batch_data_samples['sup_domain']))
            if self.local_iter >= self.burn_up_iters:
                losses.update(**self.model.loss(multi_batch_inputs, multi_batch_data_samples))
            self.local_iter += 1

        elif self.train_cfg.detector_cfg.get('type') in ['SemiBaseDift']:
            losses.update(**self.model.loss_by_gt_instances_strong(multi_batch_inputs['sup_strong'],
                                                                   multi_batch_data_samples['sup_strong']))
            losses.update(**self.model.loss_by_gt_instances_domain(multi_batch_inputs['sup_domain'],
                                                                   multi_batch_data_samples['sup_domain']))
            # if self.local_iter > self.burn_up_iters:
            # unsup_loss, student_fpn, dift_fpn = self.model.loss_dift(multi_batch_inputs, multi_batch_data_samples)
            # losses.update(**unsup_loss)
            dift_fpn = self.extract_feat_from_dift(multi_batch_inputs['sup_strong'])
            student_fpn = self.extract_feat(multi_batch_inputs['sup_strong'])
            losses.update(**self.cross_loss_dift_to_student(multi_batch_data_samples['sup_strong'], dift_fpn))

            if self.enable_feature_loss:
                if self.feature_loss_type in ['l1', 'mse', 'kl']:
                    feature_loss = dict()
                    feature_loss['feature_loss'] = self.feature_loss(student_fpn[-1], dift_fpn[-1])
                    losses.update(rename_loss_dict(str(self.feature_loss_type) + '_', feature_loss))

                if self.feature_loss_type in ['domain_classifier']:
                    feature_loss = self.domain_loss(student_fpn[-1], dift_fpn[-1])
                    losses.update(rename_loss_dict(str(self.feature_loss_type) + '_', feature_loss))

                if self.feature_loss_type in ['mutual_information_maximization']:
                    feature_loss = dict()
                    mi_score = self.mutual_information_maximization(student_fpn[-1], dift_fpn[-1])
                    feature_loss['feature_loss'] = self.feature_loss(mi_score)
                    losses.update(rename_loss_dict(str(self.feature_loss_type) + '_', feature_loss))

            self.local_iter += 1
        else:
            raise "detector type not in ['SemiBase','SoftTeacher','SemiBaseDift'] "
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.with_student:
            if self.model.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
                return self.model.teacher(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'student':
                return self.model.student(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'dift_detector':
                return self.model.dift_detector(batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.model(batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        return self.model(
            batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> (Tuple[Tensor], Tuple[Tensor]):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if not self.with_student:
            x_backbone = self.model.backbone(batch_inputs)
            x_neck = self.model.neck(x_backbone)
        else:
            x_backbone = self.model.student.backbone(batch_inputs)
            x_neck = self.model.student.neck(x_backbone)
        return x_neck

    def extract_feat_from_dift(self, batch_inputs: Tensor) -> (Tuple[Tensor], Tuple[Tensor]):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x_backbone = self.model.dift_detector.backbone(batch_inputs)
        x_neck = self.model.dift_detector.neck(x_backbone)

        return x_neck

    def cross_loss_dift_to_student(self, batch_data_samples: SampleList, dift_fpn):
        losses = dict()
        if not self.with_rpn:
            detector_loss = self.model.student.bbox_head.loss(dift_fpn, batch_data_samples)
            losses.update(rename_loss_dict('dift_to_student_cross_', detector_loss))
        else:
            proposal_cfg = self.model.student.train_cfg.get('rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(
                dift_fpn, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('dift_to_student_cross_', rpn_losses))
            roi_losses = self.model.student.roi_head.loss(dift_fpn, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict('dift_to_student_cross_', roi_losses))
        return losses

    def domain_loss(self, source_neck, target_neck):
        losses_domain = dict()
        da_s = self.domain_classifier(grad_reverse(source_neck))
        da_t = self.domain_classifier(grad_reverse(target_neck))
        da_loss = self.da_loss(da_s, da_t)
        losses_domain['da_loss'] = da_loss * self.feature_loss_weight

        return losses_domain
