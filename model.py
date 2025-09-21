from random import random
import random
from networks.wtconv.wtconv2d import WTConv2d
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math
from thop import profile


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


def generate_patches(image, scale_factors):
    """
    将输入图像划分为不同尺度的块
    :param image: (B, C, H, W) 原始图像
    :param scale_factors: list, 指定降采样比例
    :return: dict, 包含不同分辨率的 patches
    """
    B, C, H, W = image.shape
    patches_dict = {}

    for scale in scale_factors:
        downsampled = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        patch_size = downsampled.shape[-1]  # 获取降采样后的尺寸
        patches = downsampled.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # 变换维度 (B, num_patches, C, H', W')
        patches_dict[f"1/{int(1 / scale)}"] = patches

    return patches_dict


def channel_shuffle(tensor):
    B, C, H, W = tensor.shape
    channels = list(range(C))
    random.shuffle(channels)
    shuffled_image = tensor[:, channels, :, :]
    return shuffled_image


class WCA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WCA, self).__init__()
        self.WTConv = WTConv2d(in_channels, out_channels, kernel_size, stride, bias, wt_levels, wt_type)
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                              bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.WTConv(x1)
        x3 = self.conv(x2)
        result = torch.mul(x, x3)
        return result


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, bias=1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)
        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v


class MCA(nn.Module):
    def __init__(self, channels, scale=7, gamma=2, bias=1, patch_scale=[2, 4]):
        super(MCA, self).__init__()
        self.scale = scale
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.patch_scale = patch_scale
        self.patch_size_num = patch_scale[0] * patch_scale[0] + patch_scale[1] * patch_scale[1] + 1

        kernel_size1 = int(abs((math.log(21 * channels, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        Xx4 = torch.reshape(x, shape=(B, -1, H // self.patch_scale[0], W // self.patch_scale[0]))
        Xx16 = torch.reshape(x, shape=(B, -1, H // self.patch_scale[1], W // self.patch_scale[1]))

        X1 = self.avg_pool(x)
        X2 = self.avg_pool(Xx4)
        X3 = self.avg_pool(Xx16)

        X2 = X2.view(B, -1, self.patch_scale[0], 1, 1).sum(dim=2)
        X3 = X3.view(B, -1, self.patch_scale[1], 1, 1).sum(dim=2)

        X = torch.cat((X1, X2, X3), dim=1)
        X_shuffle = channel_shuffle(X)
        _, Cs, _, _ = X_shuffle.shape
        X_shuffle = X_shuffle.view(B, -1, Cs // C, 1, 1).mean(dim=2)

        x1 = self.conv1(X_shuffle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2 = self.sigmoid(x1)
        result = x * x2
        return result


class MFN(nn.Module):
    def __init__(self, in_channels):
        super(MFN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = channel_shuffle(x1)
        x2 = x1[:, 0: self.in_channels, :, :]
        x3 = x1[:, self.in_channels:, :, :]

        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        result = x2 + x3
        return result


class MFN_new(nn.Module):
    def __init__(self, in_channels):
        super(MFN_new, self).__init__()
        self.in_channels = in_channels
        self.convfirst = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.convlast = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x1 = self.convfirst(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = channel_shuffle(x1)
        x2 = x1[:, 0: self.in_channels, :, :]
        x3 = x1[:, self.in_channels:, :, :]

        x2 = self.conv3(x2)
        x2 = self.bn1(x2)

        x3 = self.conv4(x3)
        x3 = self.bn2(x3)
        result = x2 + x3
        result = self.relu(result)
        result = self.convlast(result)

        return result


class MWCA(nn.Module):
    def __init__(self, in_channels):
        super(MWCA, self).__init__()
        self.WCA = WCA(in_channels, in_channels)
        self.MCA = MCA(in_channels)

    def forward(self, x):
        x1 = self.WCA(x)
        x2 = self.MCA(x1)
        return x2


class MWCA_new(nn.Module):
    def __init__(self, in_channels):
        super(MWCA_new, self).__init__()
        # self.WCA = WCA(in_channels, in_channels)
        self.MCA = MCA(in_channels)

    def forward(self, x):
        # x1 = self.WCA(x)
        x2 = self.MCA(x)
        # result = x + x2
        return x2


class MWAN(nn.Module):
    def __init__(self, in_channels):
        super(MWAN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.MWCA = MWCA_new(in_channels)
        self.MFN = MFN_new(in_channels)

    def forward(self, x):
        x1 = self.bn1(x)
        x2 = self.MWCA(x1)
        x3 = x + x2
        x4 = self.bn2(x3)
        x5 = self.MFN(x4)
        result = x3 + x5

        return result


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.wtconv1 = WTConv2d(in_channels, in_channels)
        self.wtconv2 = WTConv2d(in_channels, in_channels, kernel_size=5)
        self.pointconv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.pointconv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.lm = nn.LayerNorm(normalized_shape=in_channels)
        self.conv_para = nn.Parameter(torch.ones(size=(1, in_channels, size, size)), requires_grad=True)
        self.wtconv_para = nn.Parameter(torch.ones(size=(1, in_channels, size, size)), requires_grad=True)

    def forward(self, x):
        x = self.lm(x.transpose(1, 3))
        x = x.transpose(3, 1)
        conv1 = self.conv1(x)
        wtconv1 = self.wtconv1(x)

        conv1 = torch.mul(conv1, self.conv_para)
        wtconv1 = torch.mul(wtconv1, self.wtconv_para)

        x1 = conv1 + wtconv1

        conv2 = self.conv2(x1)
        wtconv2 = self.wtconv2(x1)

        conv2 = torch.mul(conv2, self.conv_para)
        wtconv2 = torch.mul(wtconv2, self.wtconv_para)

        x2 = conv2 + wtconv2

        x2 = self.pointconv1(x2)
        x2 = self.relu(x2)
        result = self.pointconv2(x2)

        return result


class Encoder_single33(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder_single33, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.wtconv1 = WTConv2d(in_channels, in_channels)
        self.pointconv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.pointconv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.lm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        x = self.lm(x.transpose(1, 3))
        x = x.transpose(3, 1)
        conv1 = self.conv1(x)
        wtconv1 = self.wtconv1(x)

        x1 = conv1 + wtconv1

        x2 = self.pointconv1(x1)
        x2 = self.relu(x2)
        result = self.pointconv2(x2)

        return result


class Encoder_35(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder_35, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.wtconv1 = WTConv2d(in_channels, in_channels)
        self.wtconv2 = WTConv2d(in_channels, in_channels, kernel_size=5)
        self.pointconv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.pointconv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.lm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        x = self.lm(x.transpose(1, 3))
        x = x.transpose(3, 1)
        conv1 = self.conv1(x)
        wtconv1 = self.wtconv1(x)

        x1 = conv1 + wtconv1

        conv2 = self.conv2(x1)
        wtconv2 = self.wtconv2(x1)

        x2 = conv2 + wtconv2

        x2 = self.pointconv1(x2)
        x2 = self.relu(x2)
        result = self.pointconv2(x2)

        return result


class DownSample(nn.Module):
    def __init__(self, in_channels, maxpool=True):
        super(DownSample, self).__init__()
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # self.conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        """正常 forward，使用自身参数"""
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class PRNet(nn.Module):
    def __init__(self, in_channels=3,
                 num_classes=10,
                 dims=[64, 128, 256, 512, 1024],
                 input_size=256):
        super(PRNet, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=1, stride=1),
                                  nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)
                                  )
        self.encoder1 = Encoder(dims[0], dims[1], size=input_size)
        self.encoder2 = Encoder(dims[1], dims[2], size=input_size // 2)
        self.encoder3 = Encoder(dims[2], dims[3], size=input_size // 4)
        self.encoder4 = Encoder(dims[3], dims[4], size=input_size // 8)

        self.pm1 = DownSample(in_channels=dims[0])
        self.pm2 = DownSample(in_channels=dims[1])
        self.pm3 = DownSample(in_channels=dims[2])
        self.pm4 = DownSample(in_channels=dims[3])

        self.decoder1 = Decoder(in_channels=dims[1], out_channels=dims[0])
        self.decoder2 = Decoder(in_channels=dims[2], out_channels=dims[1])
        self.decoder3 = Decoder(in_channels=dims[3], out_channels=dims[2])
        self.decoder4 = Decoder(in_channels=dims[4], out_channels=dims[3])

        self.mwca1 = MWCA_new(dims[0])
        self.mwca2 = MWCA_new(dims[1])
        self.mwca3 = MWCA_new(dims[2])
        self.mwca4 = MWCA_new(dims[3])

        self.seghead = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        stem = self.stem(x)
        x_e1 = self.encoder1(stem)
        x_p1 = self.pm1(x_e1)

        x_e2 = self.encoder2(x_p1)
        x_p2 = self.pm2(x_e2)

        x_e3 = self.encoder3(x_p2)
        x_p3 = self.pm3(x_e3)

        x_e4 = self.encoder4(x_p3)
        x_p4 = self.pm4(x_e4)

        X4 = self.decoder4(x_p4, self.mwca4(x_e4))
        X3 = self.decoder3(X4, self.mwca3(x_e3))
        X2 = self.decoder2(X3, self.mwca2(x_e2))
        X1 = self.decoder1(X2, self.mwca1(x_e1))

        result = self.seghead(X1)
        return result


class PRNet_single33(nn.Module):
    def __init__(self, in_channels=3,
                 num_classes=10,
                 dims=[64, 128, 256, 512, 1024],
                 input_size=256):
        super(PRNet_single33, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=1, stride=1),
                                  nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)
                                  )
        self.encoder1 = Encoder_single33(dims[0], dims[1], size=input_size)
        self.encoder2 = Encoder_single33(dims[1], dims[2], size=input_size // 2)
        self.encoder3 = Encoder_single33(dims[2], dims[3], size=input_size // 4)
        self.encoder4 = Encoder_single33(dims[3], dims[4], size=input_size // 8)

        self.pm1 = DownSample(in_channels=dims[0])
        self.pm2 = DownSample(in_channels=dims[1])
        self.pm3 = DownSample(in_channels=dims[2])
        self.pm4 = DownSample(in_channels=dims[3])

        self.decoder1 = Decoder(in_channels=dims[1], out_channels=dims[0])
        self.decoder2 = Decoder(in_channels=dims[2], out_channels=dims[1])
        self.decoder3 = Decoder(in_channels=dims[3], out_channels=dims[2])
        self.decoder4 = Decoder(in_channels=dims[4], out_channels=dims[3])

        self.mwca1 = MWCA_new(dims[0])
        self.mwca2 = MWCA_new(dims[1])
        self.mwca3 = MWCA_new(dims[2])
        self.mwca4 = MWCA_new(dims[3])

        self.seghead = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        stem = self.stem(x)
        x_e1 = self.encoder1(stem)
        x_p1 = self.pm1(x_e1)

        x_e2 = self.encoder2(x_p1)
        x_p2 = self.pm2(x_e2)

        x_e3 = self.encoder3(x_p2)
        x_p3 = self.pm3(x_e3)

        x_e4 = self.encoder4(x_p3)
        x_p4 = self.pm4(x_e4)

        X4 = self.decoder4(x_p4, self.mwca4(x_e4))
        X3 = self.decoder3(X4, self.mwca3(x_e3))
        X2 = self.decoder2(X3, self.mwca2(x_e2))
        X1 = self.decoder1(X2, self.mwca1(x_e1))

        result = self.seghead(X1)
        return result


class PRNet_35(nn.Module):
    def __init__(self, in_channels=3,
                 num_classes=10,
                 dims=[64, 128, 256, 512, 1024],
                 input_size=256):
        super(PRNet_35, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=1, stride=1),
                                  nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True)
                                  )
        self.encoder1 = Encoder_35(dims[0], dims[1], size=input_size)
        self.encoder2 = Encoder_35(dims[1], dims[2], size=input_size // 2)
        self.encoder3 = Encoder_35(dims[2], dims[3], size=input_size // 4)
        self.encoder4 = Encoder_35(dims[3], dims[4], size=input_size // 8)

        self.pm1 = DownSample(in_channels=dims[0])
        self.pm2 = DownSample(in_channels=dims[1])
        self.pm3 = DownSample(in_channels=dims[2])
        self.pm4 = DownSample(in_channels=dims[3])

        self.decoder1 = Decoder(in_channels=dims[1], out_channels=dims[0])
        self.decoder2 = Decoder(in_channels=dims[2], out_channels=dims[1])
        self.decoder3 = Decoder(in_channels=dims[3], out_channels=dims[2])
        self.decoder4 = Decoder(in_channels=dims[4], out_channels=dims[3])

        self.mwca1 = MWCA_new(dims[0])
        self.mwca2 = MWCA_new(dims[1])
        self.mwca3 = MWCA_new(dims[2])
        self.mwca4 = MWCA_new(dims[3])

        self.seghead = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        stem = self.stem(x)
        x_e1 = self.encoder1(stem)
        x_p1 = self.pm1(x_e1)

        x_e2 = self.encoder2(x_p1)
        x_p2 = self.pm2(x_e2)

        x_e3 = self.encoder3(x_p2)
        x_p3 = self.pm3(x_e3)

        x_e4 = self.encoder4(x_p3)
        x_p4 = self.pm4(x_e4)

        X4 = self.decoder4(x_p4, self.mwca4(x_e4))
        X3 = self.decoder3(X4, self.mwca3(x_e3))
        X2 = self.decoder2(X3, self.mwca2(x_e2))
        X1 = self.decoder1(X2, self.mwca1(x_e1))

        result = self.seghead(X1)
        return result


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = PRNet(3, 10)
    model.to(device)
    print(model)
    a = torch.randn(18, 3, 256, 256)
    a = a.to(device)
    b, c = model(a)
    flops, params = profile(model, inputs=(a,))
    print(b.size(), c.size())

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")  # GFLOPs
    print(f"Params: {params / 1e6:.3f} M")  # M参数
