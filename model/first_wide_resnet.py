import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


BatchNorm2d = nn.BatchNorm2d
Conv2d = nn.Conv2d


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, device = torch.device(0), normalize = None, pos = 0, eot = False, lb = 2048):
        super(WideResNet, self).__init__()
        self.eot = eot
        self.pos = pos
        self.lb = lb
        self.device = device
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        shapes = [(1, 3, 32, 32), (1, 16, 32, 32), (1, 160, 32, 32), (1, 320, 16, 16), (1, 640, 8, 8)]
        self.w_shape = shapes[self.pos]
        Mu_ = torch.ones(self.w_shape).to(device)
        SD_ = torch.ones(self.w_shape).to(device)
        self.Mu_ = nn.Parameter(Mu_)
        self.SD_ = nn.Parameter(SD_)
        self.rand_weight = torch.normal(0, 1, self.w_shape).to(device)
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        
        if self.eot:
            self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)

        if self.pos == 0:
            out = x * (self.rand_weight * self.SD_ + self.Mu_)
        else:
            out = x
        out = self.conv1(out)

        if self.pos == 1:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        out = self.block1(out)

        if self.pos == 2:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        out = self.block2(out)

        if self.pos == 3:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        if self.pos == 4:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


    def set_rands(self):
        min_SD = float(self.SD_.min())
        mean_lb = (self.lb / (self.w_shape[1] * self.w_shape[2] * self.w_shape[3]))
        if min_SD < mean_lb:
            with torch.no_grad():
                self.SD_[self.SD_ < mean_lb] += mean_lb - min_SD
        if not self.eot:
            self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)



def WideResNet34(num_classes=10, normalize=None, device = torch.device(0), pos = 0, eot = False, lb = 2048):

    return WideResNet(num_classes=num_classes, normalize = normalize, pos = pos, device = device, eot = eot, lb = lb)
