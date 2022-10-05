import torch.nn as nn
import torch

from torch.nn import Parameter

class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, LR=0.01):
        super(DilatedResBlock, self).__init__()

        self.down = in_channels != out_channels
        stride = 2 if self.down else 1

        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, stride=stride, padding=1, bias=False))
        self.main.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.main.append(nn.LeakyReLU(LR, inplace=True))

        self.r_path = list()
        self.r_path.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, stride=stride, padding=1, bias=False))
        self.r_path.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.r_path.append(nn.LeakyReLU(LR, inplace=True))

        self.main = nn.Sequential(*self.main)
        self.r_path = nn.Sequential(*self.r_path)

    def forward(self, x):
        return self.main(x) + self.r_path(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, LR=0.01):
        super(ResBlock, self).__init__()

        self.down = in_channels != out_channels
        stride = 2 if self.down else 1

        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.main.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.main.append(nn.LeakyReLU(LR, inplace=True))

        self.r_path = list()
        self.r_path.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.r_path.append(nn.InstanceNorm2d(out_channels, affine=True))
        self.r_path.append(nn.LeakyReLU(LR, inplace=True))

        self.main = nn.Sequential(*self.main)
        self.r_path = nn.Sequential(*self.r_path)

    def forward(self, x):
        return self.main(x) + self.r_path(x)


class GatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        mask = self.mask(x)
        gated_mask = self.sigmoid(mask)
        return gated_mask * x


# Reference : https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)