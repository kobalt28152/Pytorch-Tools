import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation, SqueezeExcitation

from collections import OrderedDict

class ConvBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim, down=False, **params):
        super().__init__()
        self.module = nn.Sequential(
            OrderedDict([
                ('downsample', nn.MaxPool2d(kernel_size=2) if down else nn.Identity()),
                ('conv-bn-relu-1', Conv2dNormActivation(in_dim, out_dim, kernel_size=3, padding=1, **params)),
                ('conv-bn-relu-2', Conv2dNormActivation(out_dim, out_dim, kernel_size=3, padding=1, **params))
            ])
        )

    def forward(self, x):
        return self.module(x)


class ResidualV1(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, bottleneck=False, hidden_dim=None):
        super().__init__()
        if bottleneck and hidden_dim is None: hidden_dim = int(round(in_dim / 2))
        self.module = self._get_module(in_dim, out_dim, hidden_dim, stride, bottleneck)

        if (stride != 1) or (in_dim != out_dim):
            self.skip = Conv2dNormActivation(in_dim, out_dim, kernel_size=1, stride=stride, activation_layer=None)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        input = x

        x = self.module(x)
        input = self.skip(input)
        x = x + input

        return self.relu(x)

    def _get_module(self, in_dim, out_dim, hidden_dim, stride, bottleneck):
        if bottleneck:
            return nn.Sequential(
                OrderedDict([
                    ('conv-1x1-in', Conv2dNormActivation(in_dim, hidden_dim, kernel_size=1)),
                    ('conv-3x3', Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1)),
                    ('conv-1x1-out', Conv2dNormActivation(hidden_dim, out_dim, kernel_size=1, activation_layer=None)),
                ])
                )
        else:
            return nn.Sequential(
                OrderedDict([
                    ('conv-bn-relu-1', Conv2dNormActivation(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)),
                    ('conv-bn-2', Conv2dNormActivation(out_dim, out_dim, kernel_size=3, stride=1, padding=1, activation_layer=None)),
                ])
            )


class ResidualV2(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = None
        if (stride != 1) or (in_dim != out_dim):
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=stride, bias=False)

    def forward(self, x):
        input = x
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)

        if self.downsample is not None:
            input = self.downsample(input)

        x = x + input

        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=6, stride=1, **params):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'Inverted Bottleneck only accepts stride = 1 or 2; stride={stride} passed')

        hidden_dim = int(round(in_dim * expansion))
        self.use_skip = (stride == 1) and  (in_dim == out_dim)

        self.module = nn.Sequential(
            OrderedDict([
                ('conv-1x1-in', Conv2dNormActivation(in_dim, hidden_dim, kernel_size=1, **params)),
                ('conv-3x3-dw', Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                                     stride=stride, groups=hidden_dim, **params)),
                ('conv-1x1-ou', Conv2dNormActivation(hidden_dim, out_dim, kernel_size=1, **params))
            ])
        )

    def forward(self, x):
        input = x
        x = self.module(x)
        if self.use_skip:
            return x + input
        else:
            return x


class SE_InvertedResidual(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=6, stride=1, r=0.25, **params):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f'Inverted Bottleneck only accepts stride = 1 or 2; stride={stride} passed')

        hidden_dim = int(round(in_dim * expansion))
        squeeze_dim = int(round(hidden_dim * r))
        self.use_skip = (stride == 1) and  (in_dim == out_dim)

        self.module = nn.Sequential(
            OrderedDict([
                ('conv-1x1-in', Conv2dNormActivation(in_dim, hidden_dim, kernel_size=1, **params)),
                ('conv-3x3-dw', Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                                     stride=stride, groups=hidden_dim, **params)),
                ('squeeze-excitation', SqueezeExcitation(hidden_dim, squeeze_dim)),
                ('conv-1x1-ou', Conv2dNormActivation(hidden_dim, out_dim, kernel_size=1, **params))
            ])
        )

    def forward(self, x):
        input = x
        x = self.module(x)
        
        if self.use_skip:
            return x + input
        else:
            return x


# Blocks used in UNet decoder

class NNCatConv(nn.Module):
    """ Nearest Neighbor upsample + cat + convolution """
    def __init__(self, in_dim0, in_dim1, ou_dim, scale=2,
                 block=ConvBNReLU, **params):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=scale)
        self.conv = block(in_dim0+in_dim1, ou_dim, **params)
        
    def forward(self, x, f):
        x = self.up(x)
        if f is not None:
            x = torch.cat([x, f], dim=1)    # (N, C, H, W), concat along C dim
        x = self.conv(x)
        
        return x


class ConvTCatConv(nn.Module):
    """ Transposed convolution + cat + convolution """
    def __init__(self, in_dim0, in_dim1, ou_dim, scale=2, convT_den=None,
                 block=ConvBNReLU, **params):
        super().__init__()
        ou_convT = in_dim0//convT_den if convT_den is not None else in_dim0//2
        self.convT = nn.ConvTranspose2d(in_dim0, ou_convT, kernel_size=2, stride=scale)
        self.conv = block(ou_convT+in_dim1, ou_dim, **params)

    def forward(self, x, f):
        x = self.convT(x)
        if f is not None:
            x = torch.cat([x, f], dim=1)    # (N, C, H, W), concat along C dim
        x = self.conv(x)

        return x


class PShuffleCatConv(nn.Module):
    def __init__(self, in_dim0, in_dim1, ou_dim, scale=2,
                 block=ConvBNReLU, **params):
        super().__init__()
        self.conv_ps = Conv2dNormActivation(in_dim0, in_dim0*(scale**2), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.conv = block(in_dim0+in_dim1, ou_dim, **params)

    def forward(self, x, f):
        x = self.conv_ps(x)
        x = self.ps(x)

        if f is not None:
            x = torch.cat([x, f], dim=1)    # (N, C, H, W), concat along C dim
        x = self.conv(x)

        return x
