import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.base_model import SCSE, ResInception, BasicConv2d,SameConv2d,SeRes
from model.xception import xception
class Decoderv2(nn.Module):
    def __init__(self, x1_c, x2_c, out_c, pad = None):
        super(Decoderv2, self).__init__()
        self.ct_out = x1_c // 2
        self.x2_c = x2_c
        self.ct_conv = nn.ConvTranspose2d(x1_c, self.ct_out, 2, stride=2)
        self.conx2 = BasicConv2d(x2_c, self.ct_out, 3, 1)
        self.res1 = SeRes(x1_c)
        self.res2 = SeRes(x1_c)
        self.con = BasicConv2d(x1_c, out_c, 3, 1)
    def forward(self, x1, x2):
        x = self.ct_conv(x1)
        if self.x2_c != self.ct_out:
            x2 = self.conx2(x2)
        x = torch.cat([x, x2], 1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.con(x)
        return x


class Decoder(nn.Module):
    def __init__(self, x1_c, out_c):
        super(Decoder, self).__init__()
        self.ct_conv = nn.ConvTranspose2d(x1_c, out_c, 2, stride=2)
        self.res1 = SeRes(out_c, 8)
        self.res2 = SeRes(out_c, 8)
    def forward(self, x):
        x = self.ct_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# stage1 model
class AUNet(nn.Module):
    def __init__(self):
        super(AUNet, self).__init__()
        self.xception = xception()

        self.encode1 = nn.Sequential(self.xception.conv1,
                                           self.xception.bn1,
                                           self.xception.relu1,
                                           self.xception.conv2,
                                           self.xception.bn2,
                                           self.xception.relu2)
        self.encode2 = self.xception.block1
        self.encode3 = self.xception.block2
        self.encode4 = self.xception.block3
        self.encode5 = nn.Sequential(self.xception.block4,
                                           self.xception.block5,
                                           self.xception.block6,
                                           self.xception.block7,
                                           self.xception.block8,
                                           self.xception.block9,
                                           self.xception.block10,
                                           self.xception.block11)

        self.center = nn.Sequential(SeRes(728),
                                    SeRes(728),
                                    BasicConv2d(728, 256, 3, 1),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 728, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 64)

        self.logit = nn.Sequential(nn.Conv2d(4, 1, kernel_size=1, bias=False))

        self.logit1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.logit2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.logit3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.logit4 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)
        e1 = self.encode1(x)  # 64, 128, 128
        e2 = self.encode2(e1)  # 128, 64, 64
        e3 = self.encode3(e2)  # 256, 32, 32
        e4 = self.encode4(e3)  # 512, 16, 16
        e5 = self.encode5(e4)  # 512, 16, 16

        center = self.center(e5)  # 512, 8, 8

        d5 = self.decode5(center, e5)  # 256, 16, 16
        d4 = self.decode4(d5, e3)  # 128, 32, 32
        e2 = F.pad(e2, (1, 0, 1, 0), "constant", value=0)
        d3 = self.decode3(d4, e2)  # 64, 64, 64
        e1 = F.pad(e1, (3, 0, 3, 0), "constant", value=0)
        d2 = self.decode2(d3, e1)  # 32, 128, 128
        # d1 = self.decode1(d2)  # 16, 256, 256

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



