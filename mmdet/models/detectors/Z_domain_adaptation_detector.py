# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .Z_domain_aug import domain_aug
from .Z_domain_classifier import FCDiscriminator_img, da_focal_loss, grad_reverse, \
    uncertainty_focal_loss, Uncertainty_FCDiscriminator_img
from .base import BaseDetector
from ..losses import PKDLoss


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
        self.use_uda = None
        self.use_uncertainty = None
        self.apply_domain_aug = None
        self.apply_teacher_aug = None
        self.adaptive_threshold = None
        self.pkd_loss = PKDLoss()
        if self.train_cfg.da_cfg.get('use_uda', True):
            self.use_uda = True
            self.da_loss = da_focal_loss
        if self.train_cfg.da_cfg.get('use_uncertainty', True):
            self.use_uncertainty = True
            self.un_loss = uncertainty_focal_loss
        if self.train_cfg.domain_aug_cfg.get('apply_domain_aug', True):
            self.apply_domain_aug = True
        if self.train_cfg.domain_aug_cfg.get('apply_teacher_aug', True):
            self.apply_teacher_aug = True
        if self.train_cfg.domain_aug_cfg.get('adaptive_threshold', True):
            self.adaptive_threshold = True
        self.domain_aug_methods = self.train_cfg.domain_aug_cfg.get('domain_aug_methods', ['FDA', 'HM', 'PDA'])
        self.mean = data_preprocessor['data_preprocessor']['mean']
        self.std = data_preprocessor['data_preprocessor']['std']
        self.weight_source = None
        self.weight_target = None

        # img = torch.randn(1, 3, 512, 512).cuda()
        # if self.with_student:
        #     img_backbone = self.model.student.backbone(img)
        #     backbone_list = [x.size(1) for x in img_backbone]
        #     img_neck = self.model.student.neck(img_backbone)
        #     neck_list = [x.size(1) for x in img_neck]
        #     self.detector_name = detector['detector']['type']
        # else:
        #     img_backbone = self.model.backbone(img)
        #     backbone_list = [x.size(1) for x in img_backbone]
        #     img_neck = self.model.neck(img_backbone)
        #     neck_list = [x.size(1) for x in img_neck]
        # self.maxvalue = max(neck_list)
        # self.refine_level = int(len(neck_list) / 2)

        if self.use_uda or self.use_uncertainty:
            self.da_type = int(self.train_cfg.da_cfg.get('input_select'))
            self.mode = self.train_cfg.da_cfg.get('input_mode', 1)
            self.daloss_weight = self.train_cfg.da_cfg.get('daloss_weight', 1.0)
            self.da_start_iters = self.train_cfg.da_cfg.get('da_start_iters', 0)
            assert self.da_type in [0, 1, 2, 3]

            if self.da_type == 1:
                if self.use_uda:
                    self.Discriminator = FCDiscriminator_img(256)
                if self.use_uncertainty:
                    self.Uncertainty_Discriminator = Uncertainty_FCDiscriminator_img(256)

        self.burn_up_iters = self.train_cfg.detector_cfg.get('burn_up_iters', 0)
        self.exchange_iters = self.train_cfg.detector_cfg.get('exchange_iters', 0)
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
        if self.train_cfg.detector_cfg.get('type') == 'UDA':
            source_backbone, source_neck = self.extract_feat(multi_batch_inputs['source'])
            source_loss = self.subloss(source_neck, multi_batch_data_samples['source'])
            source_loss = rename_loss_dict('source_', reweight_loss_dict(source_loss, 1.0))
            losses.update(**source_loss)

            if self.use_uda and self.local_iter > self.da_start_iters:
                target_backbone, target_neck = self.extract_feat(multi_batch_inputs['target'])
                domain_loss = self.domain_loss(source_neck, target_neck)
                losses.update(**domain_loss)

            if self.use_uncertainty and self.local_iter > self.da_start_iters:
                self.domain_aug(multi_batch_inputs)
                source_backbone, source_neck = self.extract_feat(multi_batch_inputs['sup_domain'])
                target_backbone, target_neck = self.extract_feat(multi_batch_inputs['unsup_domain'])
                uncertainty_loss = self.uncertainty_loss(source_neck, target_neck)
                losses.update(**uncertainty_loss)

            self.local_iter += 1

        elif self.train_cfg.detector_cfg.get('type') in ['SemiBase', 'SoftTeacher']:

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
            losses.update(**self.model.loss_by_gt_instances(multi_batch_inputs['sup_weak'],
                                                                   multi_batch_data_samples['sup_weak']))
            if self.local_iter > self.burn_up_iters:
                losses.update(**self.model.loss_dift(multi_batch_inputs, multi_batch_data_samples))
            self.local_iter += 1
        else:
            raise "detector type not in ['UDA','SDA','SemiBase','SoftTeacher','SemiBaseDift'] "
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

        return x_backbone, x_neck

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

        return x_backbone, x_neck

    def subloss(self, neck: Tuple,
                batch_data_samples: SampleList):

        if not self.with_rpn:
            if self.detector_name in ['DeformableDETR', 'DINO']:
                head_inputs_dict = self.model.forward_transformer(neck, batch_data_samples)
                losses = self.model.bbox_head.loss(
                    **head_inputs_dict, batch_data_samples=batch_data_samples)
            else:
                losses = self.model.bbox_head.loss(neck, batch_data_samples)
        else:
            losses = dict()
            if self.model.with_rpn:
                proposal_cfg = self.model.train_cfg.get('rpn_proposal',
                                                        self.model.test_cfg.rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = self.model.rpn_head.loss_and_predict(
                    neck, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in list(keys):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            roi_losses = self.model.roi_head.loss(neck, rpn_results_list,
                                                  batch_data_samples)
            losses.update(roi_losses)
        return losses

    def cross_loss_dift(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        dift_backbone = self.model.dift_detector.backbone(batch_inputs)
        dift_neck = self.model.dift_detector.neck(dift_backbone)
        losses = dict()
        if not self.with_rpn:
            detector_loss = self.model.student.bbox_head.loss(dift_neck, batch_data_samples)
            losses.update(rename_loss_dict('dift_to_student_cross_', detector_loss))
        else:
            proposal_cfg = self.model.student.train_cfg.get('rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(
                dift_neck, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('dift_to_student_cross_', rpn_losses))
            roi_losses = self.model.student.roi_head.loss(dift_neck, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict('dift_to_student_cross_', roi_losses))
        return losses

    def cross_loss_student(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        student_backbone = self.model.dift_detector.backbone(batch_inputs)
        student_neck = self.model.dift_detector.neck(student_backbone)
        losses = dict()
        if not self.with_rpn:
            detector_loss = self.model.dift_detector.bbox_head.loss(student_neck, batch_data_samples)
            losses.update(rename_loss_dict('student_to_dift_cross_', detector_loss))
        else:
            proposal_cfg = self.model.dift_detector.train_cfg.get('rpn_proposal', self.model.dift_detector.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.dift_detector.rpn_head.loss_and_predict(
                student_neck, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('student_to_dift_cross_', rpn_losses))
            roi_losses = self.model.dift_detector.roi_head.loss(student_neck, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict('student_to_dift_cross_', roi_losses))
        return losses

    def domain_loss(self, source_neck, target_neck):
        losses_domain = dict()
        if self.da_type == 1:
            da_s = self.Discriminator(grad_reverse(source_neck[-1]))
            da_t = self.Discriminator(grad_reverse(target_neck[-1]))
            da_loss = self.da_loss(da_s, da_t)
            losses_domain['da_loss'] = da_loss
            domain_loss = rename_loss_dict(self.mode[self.da_type] + '_',
                                           reweight_loss_dict(losses_domain, self.daloss_weight))

        else:
            raise "discriminator not in  ['backbone[-1]', 'neck[-1]', 'neck_all',  'neck_cat' ]"

        return domain_loss

    def uncertainty_loss(self, source_backbone, target_backbone):
        losses_uncertainty = dict()
        if self.da_type == 1:
            uncertainty_s = self.Uncertainty_Discriminator(grad_reverse(source_backbone[-1]))
            uncertainty_t = self.Uncertainty_Discriminator(grad_reverse(target_backbone[-1]))

            uncertainty_s_label = torch.tensor(self.weight_source, requires_grad=True,
                                               device=uncertainty_s.device)
            uncertainty_t_label = 1 - torch.tensor(self.weight_target, requires_grad=True,
                                                   device=uncertainty_t.device)

            uncertainty_loss = self.un_loss(uncertainty_s, uncertainty_t, uncertainty_s_label, uncertainty_t_label)

            losses_uncertainty['uncertainty_loss'] = uncertainty_loss
            uncertainty_loss = rename_loss_dict('uncertainty_',
                                                reweight_loss_dict(losses_uncertainty, self.daloss_weight))
        else:
            raise "discriminator not in  ['backbone[-1]', 'neck[-1]', 'neck_all',  'neck_cat' ]"

        return uncertainty_loss

    def neck_address(self, neck):
        feats = []
        gather_size = neck[self.refine_level].size()[2:]
        for i in range(len(neck)):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    neck[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    neck[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        feature = torch.cat(feats, dim=1)
        return feature

    def domain_aug(self, multi_batch_inputs):

        source_domain = multi_batch_inputs['sup_domain'].clone().permute(0, 2, 3, 1)
        target_domain = multi_batch_inputs['unsup_domain'].clone().permute(0, 2, 3, 1)

        source_domain[:, :, :, 0] = source_domain[:, :, :, 0] * self.std[0] + self.mean[0]
        source_domain[:, :, :, 1] = source_domain[:, :, :, 1] * self.std[1] + self.mean[1]
        source_domain[:, :, :, 2] = source_domain[:, :, :, 2] * self.std[2] + self.mean[2]

        target_domain[:, :, :, 0] = target_domain[:, :, :, 0] * self.std[0] + self.mean[0]
        target_domain[:, :, :, 1] = target_domain[:, :, :, 1] * self.std[1] + self.mean[1]
        target_domain[:, :, :, 2] = target_domain[:, :, :, 2] * self.std[2] + self.mean[2]

        source_list = [np.asarray(source_domain[index, :, :, :].cpu().clone(), dtype=np.uint8) for
                       index in range(source_domain.shape[0])]
        target_list = [np.asarray(target_domain[index, :, :, :].cpu().clone(), dtype=np.uint8) for
                       index in range(target_domain.shape[0])]

        self.weight_source = random.sample(np.arange(0.2, 0.8, 0.05).tolist(), len(source_list))
        self.weight_target = random.sample(np.arange(0.2, 0.8, 0.05).tolist(), len(target_list))
        # weight = self.local_iter%1000/1000.0
        # self.weight_source = np.array([weight]*len(source_list))
        # self.weight_target = np.array([weight]*len(target_list))
        for index in range(len(source_list)):
            source_domain[index, :, :, :] = torch.tensor(
                domain_aug(source_list[index], random.choice(target_list), blend_ratio=self.weight_source[index],
                           domain_aug_methods=self.domain_aug_methods))

        for index in range(len(target_list)):
            target_domain[index, :, :, :] = torch.tensor(
                domain_aug(target_list[index], random.choice(source_list), blend_ratio=self.weight_target[index],
                           domain_aug_methods=self.domain_aug_methods))

        source_domain[:, :, :, 0] = (source_domain[:, :, :, 0] - self.mean[0]) / self.std[0]
        source_domain[:, :, :, 1] = (source_domain[:, :, :, 1] - self.mean[1]) / self.std[1]
        source_domain[:, :, :, 2] = (source_domain[:, :, :, 2] - self.mean[2]) / self.std[2]

        multi_batch_inputs['sup_domain'].data = source_domain.permute(0, 3, 1, 2)

        target_domain[:, :, :, 0] = (target_domain[:, :, :, 0] - self.mean[0]) / self.std[0]
        target_domain[:, :, :, 1] = (target_domain[:, :, :, 1] - self.mean[1]) / self.std[1]
        target_domain[:, :, :, 2] = (target_domain[:, :, :, 2] - self.mean[2]) / self.std[2]

        multi_batch_inputs['unsup_domain'].data = target_domain.permute(0, 3, 1, 2)

    def domain_mix(self, multi_batch_inputs):

        source_domain = multi_batch_inputs['sup_mix'].clone().permute(0, 2, 3, 1)
        target_domain = multi_batch_inputs['unsup_mix'].clone().permute(0, 2, 3, 1)

        source_domain[:, :, :, 0] = source_domain[:, :, :, 0] * self.std[0] + self.mean[0]
        source_domain[:, :, :, 1] = source_domain[:, :, :, 1] * self.std[1] + self.mean[1]
        source_domain[:, :, :, 2] = source_domain[:, :, :, 2] * self.std[2] + self.mean[2]

        target_domain[:, :, :, 0] = target_domain[:, :, :, 0] * self.std[0] + self.mean[0]
        target_domain[:, :, :, 1] = target_domain[:, :, :, 1] * self.std[1] + self.mean[1]
        target_domain[:, :, :, 2] = target_domain[:, :, :, 2] * self.std[2] + self.mean[2]

        source_list = [np.asarray(source_domain[index, :, :, :].cpu().clone(), dtype=np.uint8) for
                       index in range(source_domain.shape[0])]
        target_list = [np.asarray(target_domain[index, :, :, :].cpu().clone(), dtype=np.uint8) for
                       index in range(target_domain.shape[0])]

        self.weight_mix_source = random.sample(np.arange(0.5, 0.9, 0.05).tolist(), len(source_list))
        self.weight_mix_target = random.sample(np.arange(0.5, 0.9, 0.05).tolist(), len(source_list))

        for index in range(len(source_list)):
            source_domain[index, :, :, :] = torch.tensor(
                cv2.addWeighted(source_list[index], self.weight_mix_source[index],
                                random.choice(target_list), 1 - self.weight_mix_source[index], 0))

        for index in range(len(target_list)):
            target_domain[index, :, :, :] = torch.tensor(
                cv2.addWeighted(target_list[index], self.weight_mix_target[index],
                                random.choice(source_list), 1 - self.weight_mix_target[index], 0))

        source_domain[:, :, :, 0] = (source_domain[:, :, :, 0] - self.mean[0]) / self.std[0]
        source_domain[:, :, :, 1] = (source_domain[:, :, :, 1] - self.mean[1]) / self.std[1]
        source_domain[:, :, :, 2] = (source_domain[:, :, :, 2] - self.mean[2]) / self.std[2]

        multi_batch_inputs['sup_mix'].data = source_domain.permute(0, 3, 1, 2)

        target_domain[:, :, :, 0] = (target_domain[:, :, :, 0] - self.mean[0]) / self.std[0]
        target_domain[:, :, :, 1] = (target_domain[:, :, :, 1] - self.mean[1]) / self.std[1]
        target_domain[:, :, :, 2] = (target_domain[:, :, :, 2] - self.mean[2]) / self.std[2]

        multi_batch_inputs['unsup_mix'].data = target_domain.permute(0, 3, 1, 2)
