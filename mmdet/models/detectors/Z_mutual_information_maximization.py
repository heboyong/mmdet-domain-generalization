import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variance."""
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

class MIEstimator(nn.Module):
    def __init__(self, input_dim):
        super(MIEstimator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, y):

        # Global average pooling to reduce the spatial dimensions
        x = norm(x)
        y = norm(y)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # [N, reduced_dim]
        y = F.adaptive_avg_pool2d(y, (1, 1)).view(y.size(0), -1)  # [N, reduced_dim]
        # Concatenate x and y along the feature dimension
        xy = torch.cat([x, y], dim=1)  # [N, reduced_dim*2]
        # Pass through the fully connected layers
        return self.fc(xy)


class MIMaxLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MIMaxLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, mi_scores, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        total_loss = sum(-mi_score.mean() for mi_score in mi_scores)

        if reduction == 'mean':
            total_loss = total_loss.mean()
        elif reduction == 'sum':
            total_loss = total_loss.sum()

        total_loss = torch.log(1 + torch.abs(total_loss))
        total_loss = self.loss_weight * total_loss

        return total_loss
