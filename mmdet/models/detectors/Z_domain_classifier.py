import torch
import torch.nn.functional as F
from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCDiscriminator_img(nn.Module):
    def __init__(self, input_ch):
        self.out_ch = 256
        super(FCDiscriminator_img, self).__init__()
        self.conv1 = conv3x3(input_ch, self.out_ch, stride=2)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.conv2 = conv3x3(self.out_ch, self.out_ch, stride=2)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        self.conv3 = conv3x3(self.out_ch, self.out_ch, stride=2)
        self.bn3 = nn.BatchNorm2d(self.out_ch)
        self.fc = nn.Linear(self.out_ch, 1)
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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, int(self.out_ch))
        x = self.fc(x)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


def da_focal_loss(f_s, f_t):
    if f_s.dim() > 2:
        f_s = f_s.mean(dim=[2, 3])
        f_t = f_t.mean(dim=[2, 3])
    s_label = torch.zeros_like(f_s, requires_grad=True, device=f_s.device)
    t_label = torch.ones_like(f_t, requires_grad=True, device=f_t.device)
    weight_s = f_t.shape[0] / (f_t.shape[0] + f_s.shape[0])
    weight_t = f_s.shape[0] / (f_t.shape[0] + f_s.shape[0])
    daloss_s = sigmoid_focal_loss_jit(f_s, s_label, gamma=3.0, reduction='mean')
    daloss_t = sigmoid_focal_loss_jit(f_t, t_label, gamma=3.0, reduction='mean')
    return daloss_s * weight_s + daloss_t * weight_t

