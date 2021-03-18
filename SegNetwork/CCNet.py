import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from .cc import CC_module as CrissCrossAttention

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        inc_feature = self.inc(x)

        res_1 = self.res1(inc_feature)
        res_1 = res_1 + inc_feature

        down1 = self.down1(res_1)

        res_2 = self.res2(down1)
        res_2 = res_2 + down1

        down2 = self.down2(res_2)

        res_3 = self.res3(down2)
        res_3 = res_3 + down2

        down3 = self.down3(res_3)

        return down3
    

class RCCmodule(nn.Module):
    def __init__(self):
        super(RCCmodule, self).__init__()

        self.conva = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=False)
                                   )
        self.cca = CrissCrossAttention(128)

        self.convb = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=False)
                                   )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(4, 128), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,x,recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output
    
class SegNetwork(nn.module):
    def __init__(self):
        super(SegNetwork, self).__init__()

        self.cnn = ResNet()
        self.seg = RCCmodule()

    def forward(self,x):

        out = self.cnn(x)
        out = self.seg(out)

        return out

