# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .utils import weighted_loss
import torch
import torch.nn.functional as F
from torch import nn


def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variance."""
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


def calculate_losses(pred, target, loss_types):
    pred = norm(pred)
    target = norm(target)
    losses = []
    if "mse" in loss_types:
        mse_loss = F.mse_loss(pred, target, reduction='none') / 2
        losses.append(mse_loss)
    if "l1" in loss_types:
        l1_loss = F.l1_loss(pred, target, reduction='none')
        losses.append(l1_loss)
    if "kl" in loss_types:
        pred_log_softmax = F.log_softmax(pred, dim=1)
        target_softmax = F.softmax(target, dim=1)
        kl_div_loss = F.kl_div(pred_log_softmax, target_softmax, reduction='none')
        losses.append(kl_div_loss)
    if losses:
        total_loss = sum(losses)
    else:
        raise ValueError("At least one loss type must be selected.")
    return total_loss


@MODELS.register_module()
class KDLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, loss_types=None):
        super(KDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_types = loss_types if loss_types is not None else "mse"

        # Validate loss_types
        assert self.loss_types in ['l1', 'mse', 'kl']

    def forward(self,
                pred,
                target,
                reduction_override=None) -> torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        losses = dict()
        reduction = reduction_override if reduction_override else self.reduction
        total_loss = calculate_losses(pred, target, self.loss_types)
        if reduction == 'mean':
            total_loss = total_loss.mean()
        elif reduction == 'sum':
            total_loss = total_loss.sum()
        total_loss = self.loss_weight * total_loss
        return total_loss
