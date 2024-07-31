import torch
import torch.nn as nn
import torch.nn.functional as F


class MIEstimator(nn.Module):
    def __init__(self, input_dim):
        super(MIEstimator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._init_weights()

    def _init_weights(self, init_method='kaiming'):
        def kaiming_init(module,
                         a=0,
                         mode='fan_out',
                         nonlinearity='relu',
                         bias=0,
                         distribution='normal'):
            assert distribution in ['uniform', 'normal']
            if distribution == 'uniform':
                nn.init.kaiming_uniform_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                nn.init.kaiming_normal_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        def xavier_init(module, gain=1, bias=0, distribution='normal'):
            assert distribution in ['uniform', 'normal']
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        if init_method == 'kaiming':
            kaiming_init(self.conv1)
            kaiming_init(self.conv2)
            kaiming_init(self.conv3)
        else:
            xavier_init(self.conv1)
            xavier_init(self.conv2)
            xavier_init(self.conv3)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        xy = torch.cat([x, y], dim=1)
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

        total_loss = self.loss_weight * total_loss

        return total_loss
