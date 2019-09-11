import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.base_model import SCSE, ResInception, BasicConv2d,SameConv2d,SeRes
class AUNet(nn.Module):
    def __init__(self):
        super(AUNet, self).__init__()

        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1,
                                     SCSE(64))
        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCSE(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCSE(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCSE(512))
        self.center = down(512, 512)

        self.d5 = up(512, 768, 256)
        self.d4 = up(256, 384, 128)
        self.d3 = up(128, 192, 64)
        self.d2 = up(64, 96, 32)

        self.logit = nn.Sequential(nn.Conv2d(4, 1, kernel_size=1, bias=False))

        self.logit1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.logit2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.logit3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, bias=False))
        self.logit4 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)
        x = self.conv1(x)  # 64, 128, 128
        e2 = self.encode2(x)  # 64, 128, 128
        e3 = self.encode3(e2)  # 128, 64, 64
        e4 = self.encode4(e3)  # 256, 32, 32
        e5 = self.encode5(e4)  # 512, 16, 16

        center = self.center(e5)  # 512, 8, 8

        d5 = self.d5(center, e5)  # 256, 16, 16
        d4 = self.d4(d5, e4)  # 128, 32, 32
        d3 = self.d3(d4, e3)  # 64, 64, 64
        d2 = self.d2(d3, e2)  # 32, 128, 128
        # # d1 = self.decode1(d2)  # 16, 256, 256
        #
        logit1 = self.logit1(d2)
        logit2 = self.logit2(F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True))
        logit3 = self.logit3(F.upsample(d4, scale_factor=4, mode='bilinear', align_corners=True))
        logit4 = self.logit4(F.upsample(d5, scale_factor=8, mode='bilinear', align_corners=True))

        f = torch.cat((logit1,
                       logit2,
                       logit3,
                       logit4), 1)  # 320, 256, 256
        logit = self.logit(f)  # 1, 256, 256
        return logit, logit1, logit2, logit3, logit4
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, in_ch2, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x1


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x