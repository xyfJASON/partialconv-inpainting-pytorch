import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models


def init_weights(init_type=None, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type is None:
                m.reset_parameters()
            else:
                raise ValueError(f'invalid initialization method: {init_type}.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


class PartialConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 norm: str = None,
                 activation: str = None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros((out_channels, )))

        mask_conv_weight = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        self.register_buffer('mask_conv_weight', mask_conv_weight)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        self.conv.apply(init_weights('kaiming'))

    def forward(self, X: Tensor, mask: Tensor):
        """ Note that 1 in mask denotes valid (unmasked) pixels,
            and 0 in mask denotes invalid pixels (holes).

        Args:
            X: Tensor[bs, C, H, W]
            mask: Tensor[bs, C, H, W]
        """
        with torch.no_grad():
            update_mask = F.conv2d(mask, self.mask_conv_weight, stride=self.stride, padding=self.padding)
            scale = self.in_channels * self.kernel_size * self.kernel_size / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0, 1)
            scale = scale * update_mask
        X = self.conv(X * mask)
        X = X * scale + self.bias.view(1, -1, 1, 1)
        X = X * update_mask
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)
        return X, update_mask


class TransposePartialConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 scale_factor: float = 2,
                 norm: str = None,
                 activation: str = None):
        super().__init__()
        self.partial_conv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding, norm, activation)
        self.scale_factor = scale_factor

    def forward(self, X: Tensor, mask: Tensor, X_skips: Tensor, mask_skips: Tensor):
        X = F.interpolate(X, scale_factor=self.scale_factor, mode='nearest')
        mask = F.interpolate(mask, scale_factor=self.scale_factor, mode='nearest')
        X, mask = self.partial_conv(torch.cat([X, X_skips], dim=1), torch.cat([mask, mask_skips], dim=1))
        return X, mask


class Generator(nn.Module):
    def __init__(self, n_layers: int = 8, freeze_enc_bn: bool = False):
        super().__init__()
        assert n_layers >= 4, f'n_layers must >= 4, get {n_layers}'
        self.n_layers = n_layers
        self.freeze_enc_bn = freeze_enc_bn

        self.encoder1 = PartialConv2d(3, 64, kernel_size=7, stride=2, padding=3, activation='relu')                 # 1/2
        self.encoder2 = PartialConv2d(64, 128, kernel_size=5, stride=2, padding=2, norm='bn', activation='relu')    # 1/4
        self.encoder3 = PartialConv2d(128, 256, kernel_size=5, stride=2, padding=2, norm='bn', activation='relu')   # 1/8
        self.encoder4 = PartialConv2d(256, 512, kernel_size=3, stride=2, padding=1, norm='bn', activation='relu')   # 1/16
        for i in range(5, n_layers+1):
            setattr(self, f'encoder{i}', PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, norm='bn', activation='relu'))
            setattr(self, f'decoder{i}', TransposePartialConv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu'))
        self.decoder4 = TransposePartialConv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')   # 1/8
        self.decoder3 = TransposePartialConv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')   # 1/4
        self.decoder2 = TransposePartialConv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')     # 1/2
        self.decoder1 = TransposePartialConv2d(64 + 3, 3, kernel_size=3, stride=1, padding=1, scale_factor=2)                                           # 1/1

    def forward(self, X: Tensor, mask: Tensor):
        X_skips, mask_skips = [X], [mask]
        for i in range(1, self.n_layers+1):
            layer = getattr(self, f'encoder{i}')
            X, mask = layer(X, mask)
            X_skips.append(X)
            mask_skips.append(mask)
        for i in range(self.n_layers, 0, -1):
            layer = getattr(self, f'decoder{i}')
            X, mask = layer(X, mask, X_skips[i-1], mask_skips[i-1])
        return X

    def train(self, mode=True):
        """
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        if not self.freeze_enc_bn:
            return
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'encoder' in name:
                module.eval()


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg16 = models.vgg16(pretrained=True)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.pool1 = nn.Sequential(*vgg16.features[:5])
        self.pool2 = nn.Sequential(*vgg16.features[5:10])
        self.pool3 = nn.Sequential(*vgg16.features[10:17])
        for i in range(1, 4):
            for param in getattr(self, f'pool{i}').parameters():
                param.requires_grad_(False)

    def forward(self, X: Tensor):
        pool1 = self.pool1(X)
        pool2 = self.pool2(pool1)
        pool3 = self.pool3(pool2)
        return pool1, pool2, pool3


def _test():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.overhead import count_params, calc_flops, calc_inference_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(n_layers=8).to(device).eval()
    X = torch.randn(1, 3, 512, 512).to(device)
    mask = torch.randint(0, 2, size=(1, 3, 512, 512)).float().to(device)
    count_params(G)
    print('=' * 60)
    calc_flops(G, dummy_input=(X, mask))
    print('=' * 60)
    calc_inference_time(G, dummy_input=(X, mask))

    G = Generator(n_layers=8, freeze_enc_bn=True).to(device)
    G.train()


if __name__ == '__main__':
    _test()
