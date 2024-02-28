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
        if input_ch != 256:
            self.conv1 = conv3x3(input_ch, int(input_ch / 2), stride=1)
            self.bn1 = nn.BatchNorm2d(int(input_ch / 2))
            self.conv2 = conv3x3(int(input_ch / 2), int(input_ch / 4), stride=1)
            self.bn2 = nn.BatchNorm2d(int(input_ch / 4))
            self.conv3 = conv3x3(int(input_ch / 4), self.out_ch, stride=1)
            self.bn3 = nn.BatchNorm2d(self.out_ch)
            self.fc = nn.Linear(self.out_ch, 1)
        else:
            self.conv1 = conv3x3(input_ch, self.out_ch, stride=1)
            self.bn1 = nn.BatchNorm2d(self.out_ch)
            self.conv2 = conv3x3(self.out_ch, self.out_ch, stride=1)
            self.bn2 = nn.BatchNorm2d(self.out_ch)
            self.conv3 = conv3x3(self.out_ch, self.out_ch, stride=1)
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


class Uncertainty_FCDiscriminator_img(nn.Module):
    def __init__(self, input_ch):
        self.out_ch = 256
        super(Uncertainty_FCDiscriminator_img, self).__init__()
        if input_ch != 256:
            self.conv1 = conv3x3(input_ch, int(input_ch / 2), stride=1)
            self.bn1 = nn.BatchNorm2d(int(input_ch / 2))
            self.conv2 = conv3x3(int(input_ch / 2), int(input_ch / 4), stride=1)
            self.bn2 = nn.BatchNorm2d(int(input_ch / 4))
            self.conv3 = conv3x3(int(input_ch / 4), self.out_ch, stride=1)
            self.bn3 = nn.BatchNorm2d(self.out_ch)
            self.fc = nn.Linear(self.out_ch, 1)
        else:
            self.conv1 = conv3x3(input_ch, self.out_ch, stride=1)
            self.bn1 = nn.BatchNorm2d(self.out_ch)
            self.conv2 = conv3x3(self.out_ch, self.out_ch, stride=1)
            self.bn2 = nn.BatchNorm2d(self.out_ch)
            self.conv3 = conv3x3(self.out_ch, self.out_ch, stride=1)
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

class Mix_FCDiscriminator_img(nn.Module):
    def __init__(self, input_ch):
        self.out_ch = 256
        super(Mix_FCDiscriminator_img, self).__init__()
        if input_ch != 256:
            self.conv1 = conv3x3(input_ch, int(input_ch / 2), stride=1)
            self.bn1 = nn.BatchNorm2d(int(input_ch / 2))
            self.conv2 = conv3x3(int(input_ch / 2), int(input_ch / 4), stride=1)
            self.bn2 = nn.BatchNorm2d(int(input_ch / 4))
            self.conv3 = conv3x3(int(input_ch / 4), self.out_ch, stride=1)
            self.bn3 = nn.BatchNorm2d(self.out_ch)
            self.fc = nn.Linear(self.out_ch, 1)
        else:
            self.conv1 = conv3x3(input_ch, self.out_ch, stride=1)
            self.bn1 = nn.BatchNorm2d(self.out_ch)
            self.conv2 = conv3x3(self.out_ch, self.out_ch, stride=1)
            self.bn2 = nn.BatchNorm2d(self.out_ch)
            self.conv3 = conv3x3(self.out_ch, self.out_ch, stride=1)
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
class Soft_FCDiscriminator_img(nn.Module):
    def __init__(self, input_ch):
        self.out_ch = 256
        super(Soft_FCDiscriminator_img, self).__init__()
        if input_ch != 256:
            self.conv1 = conv3x3(input_ch, int(input_ch / 2), stride=1)
            self.bn1 = nn.BatchNorm2d(int(input_ch / 2))
            self.conv2 = conv3x3(int(input_ch / 2), int(input_ch / 4), stride=1)
            self.bn2 = nn.BatchNorm2d(int(input_ch / 4))
            self.conv3 = conv3x3(int(input_ch / 4), self.out_ch, stride=1)
            self.bn3 = nn.BatchNorm2d(self.out_ch)
            self.domain = nn.Linear(self.out_ch, 1)
            self.uncertainty = nn.Linear(self.out_ch, 1)
        else:
            self.conv1 = conv3x3(input_ch, self.out_ch, stride=1)
            self.bn1 = nn.BatchNorm2d(self.out_ch)
            self.conv2 = conv3x3(self.out_ch, self.out_ch, stride=1)
            self.bn2 = nn.BatchNorm2d(self.out_ch)
            self.conv3 = conv3x3(self.out_ch, self.out_ch, stride=1)
            self.bn3 = nn.BatchNorm2d(self.out_ch)
            self.domain = nn.Linear(self.out_ch, 1)
            self.uncertainty = nn.Linear(self.out_ch, 1)
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
        domain = self.domain(x)
        uncertainty = self.uncertainty(x)
        return domain, uncertainty


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


def uncertainty_focal_loss(uncertainty_s, uncertainty_t, uncertainty_s_label, uncertainty_t_label):
    uncertainty_s_label = torch.reshape(uncertainty_s_label, shape=uncertainty_s.shape)
    uncertainty_t_label = torch.reshape(uncertainty_t_label, shape=uncertainty_t.shape)

    weight_s = uncertainty_t.shape[0] / (uncertainty_s.shape[0] + uncertainty_t.shape[0])
    weight_t = uncertainty_s.shape[0] / (uncertainty_s.shape[0] + uncertainty_t.shape[0])

    uncertainty_loss_s = sigmoid_focal_loss_jit(uncertainty_s, uncertainty_s_label, gamma=3.0, reduction='mean')
    uncertainty_loss_t = sigmoid_focal_loss_jit(uncertainty_t, uncertainty_t_label, gamma=3.0, reduction='mean')
    return uncertainty_loss_s * weight_s + uncertainty_loss_t * weight_t


def da_cross_entropy_loss(f_s, f_t):
    if f_s.dim() > 2:
        f_s = f_s.mean(dim=[2, 3])
        f_t = f_t.mean(dim=[2, 3])
    s_label = torch.zeros_like(f_s, requires_grad=True, device=f_s.device)
    t_label = torch.ones_like(f_t, requires_grad=True, device=f_t.device)
    weight_s = f_t.shape[0] / (f_t.shape[0] + f_s.shape[0])
    weight_t = f_t.shape[0] / (f_t.shape[0] + f_s.shape[0])
    daloss_s = F.binary_cross_entropy_with_logits(f_s, s_label)
    daloss_t = F.binary_cross_entropy_with_logits(f_t, t_label)
    return daloss_s * weight_s + daloss_t * weight_t
