import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast


def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if(self.se):
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.conv3 = conv1x1x1(planes, planes//16)
            self.conv4 = conv1x1x1(planes//16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if(self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, se=False, **kwargs):
        super(ResNet18, self).__init__()
        in_channels = kwargs['in_channels']
        self.low_rate = kwargs['low_rate']
        self.alpha = kwargs['alpha']
        self.t2s_mul = kwargs['t2s_mul']
        self.base_channel = kwargs['base_channel']
        self.inplanes = (self.base_channel + self.base_channel//self.alpha*self.t2s_mul) if self.low_rate else self.base_channel // self.alpha
        self.conv1 = nn.Conv3d(in_channels, self.base_channel // (1 if self.low_rate else self.alpha),
                               kernel_size=(5, 7, 7),
                               stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.base_channel // (1 if self.low_rate else self.alpha))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.se = se
        self.layer1 = self._make_layer(block, self.base_channel // (1 if self.low_rate else self.alpha), layers[0])
        self.layer2 = self._make_layer(block, 2 * self.base_channel // (1 if self.low_rate else self.alpha), layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.base_channel // (1 if self.low_rate else self.alpha), layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.base_channel // (1 if self.low_rate else self.alpha), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.low_rate:
            self.bn2 = nn.BatchNorm1d(8*self.base_channel + 8*self.base_channel//self.alpha*self.t2s_mul)
        elif self.t2s_mul == 0:
            self.bn2 = nn.BatchNorm1d(16*self.base_channel//self.alpha)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        self.inplanes += self.low_rate * block.expansion * planes // self.alpha * self.t2s_mul

        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFM, self).__init__()
        self.layer1 = nn.Sequential(
            conv1x1x1(in_channel, out_channel),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.local_att_layer = nn.Sequential(
            conv1x1x1(out_channel, out_channel//4),
            nn.BatchNorm3d(out_channel//4),
            nn.ReLU(inplace=True),
            conv1x1x1(out_channel//4, out_channel),
            nn.BatchNorm3d(out_channel)
        )
        self.global_att_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1x1(out_channel, out_channel//4),
            nn.BatchNorm3d(out_channel//4),
            nn.ReLU(inplace=True),
            conv1x1x1(out_channel//4, out_channel),
            nn.BatchNorm3d(out_channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.layer1(x)
        local_att = self.local_att_layer(x)
        global_att = self.global_att_layer(x)
        y = y + x * self.sigmoid(local_att + global_att)
        return y


class HighRateBranch(ResNet18):
    def __init__(self, block, layers, se, **kargs):
        super().__init__(block, layers, se, **kargs)
        self.alpha = kargs['alpha']
        self.beta = kargs['beta']
        self.base_channel = kargs['base_channel']
        ksize = self.beta * 2 - 1
        self.l_maxpool = nn.Conv3d(self.base_channel//self.alpha, self.base_channel//self.alpha*self.t2s_mul,
                                   kernel_size=(ksize, 1, 1), stride=(self.beta, 1, 1), bias=False, padding=(ksize//2, 0, 0))
        self.l_layer1 = nn.Conv3d(self.base_channel//self.alpha, self.base_channel//self.alpha*self.t2s_mul,
                                  kernel_size=(ksize, 1, 1), stride=(self.beta, 1, 1), bias=False, padding=(ksize//2, 0, 0))
        self.l_layer2 = nn.Conv3d(2*self.base_channel//self.alpha, 2*self.base_channel//self.alpha*self.t2s_mul,
                                  kernel_size=(ksize, 1, 1), stride=(self.beta, 1, 1), bias=False, padding=(ksize//2, 0, 0))
        self.l_layer3 = nn.Conv3d(4*self.base_channel//self.alpha, 4*self.base_channel//self.alpha*self.t2s_mul,
                                  kernel_size=(ksize, 1, 1), stride=(self.beta, 1, 1), bias=False, padding=(ksize//2, 0, 0))
        self.l_layer4 = nn.Conv3d(8*self.base_channel//self.alpha, 8*self.base_channel//self.alpha*self.t2s_mul,
                                  kernel_size=(ksize, 1, 1), stride=(self.beta, 1, 1), bias=False, padding=(ksize//2, 0, 0))
        self.init_params()

    def forward(self, x):
        laterals = []
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (b, 8, 145, 22, 22)
        laterals.append(self.l_maxpool(x)) # (b, 8, 30, 22, 22)

        x = self.layer1(x) # (b, 8, 145, 22, 22)
        laterals.append(self.l_layer1(x)) # (b, 16, 30, 22, 22)

        x = self.layer2(x) # (b, 16, 145, 11, 11)
        laterals.append(self.l_layer2(x)) # (b, 32, 30, 11, 11)

        x = self.layer3(x) # (b, 32, 145, 6, 6)
        laterals.append(self.l_layer3(x)) # (b, 64, 30, 6, 6)

        x = self.layer4(x) # (b, 64, 145, 3, 3)
        laterals.append(self.l_layer4(x))

        return x, laterals


class LowRateBranch(ResNet18):
    def __init__(self, block, layers, se, **kargs):
        super().__init__(block, layers, se, **kargs)
        self.base_channel = kargs['base_channel']
        self.alpha = kargs['alpha']
        self.mfm1 = MFM(in_channel=self.base_channel + self.base_channel // self.alpha * self.t2s_mul,
                        out_channel=self.base_channel)
        self.mfm2 = MFM(in_channel=self.base_channel + self.base_channel // self.alpha * self.t2s_mul,
                        out_channel=self.base_channel)
        self.mfm3 = MFM(in_channel=2 * self.base_channel + 2 * self.base_channel // self.alpha * self.t2s_mul,
                        out_channel=2*self.base_channel)
        self.mfm4 = MFM(in_channel=4 * self.base_channel + 4 * self.base_channel // self.alpha * self.t2s_mul,
                        out_channel=4*self.base_channel)
        self.init_params()

    def forward(self, x, laterals):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (b, 64, 30, 22, 22)

        x = self.mfm1(laterals[0], x)
        x = torch.cat([x, laterals[0]], dim=1) # (b, 80, 30, 22, 22)
        x = self.layer1(x) # (b, 64, 30, 22, 22)

        x = self.mfm2(laterals[1], x)
        x = torch.cat([x, laterals[1]], dim=1) # (b, 80, 30, 22, 22)
        x = self.layer2(x) # (b, 128, 30, 11, 11)

        x = self.mfm3(laterals[2], x)
        x = torch.cat([x, laterals[2]], dim=1) # (b, 160, 30, 11, 11)
        x = self.layer3(x) # (2, 256, 30, 6, 6)

        x = self.mfm4(laterals[3], x)
        x = torch.cat([x, laterals[3]], dim=1) # (b, 320, 30, 6, 6)
        x = self.layer4(x) # (b, 512, 30, 3, 3)

        x = torch.cat([x, laterals[4]], dim=1) # (b, 640, 30, 3, 3)
        x = self.avgpool(x) # (b, 640, 30, 1, 1)

        x = x.transpose(1, 2).contiguous() # (b, 30, 640, 1, 1)
        x = x.view(-1, x.size(2)) # (b*30, 640)
        x = self.bn2(x)

        return x


class MultiBranchNet(nn.Module):
    def __init__(self, args):
        super(MultiBranchNet, self).__init__()
        self.args = args
        self.low_rate_branch = LowRateBranch(block=BasicBlock,
                                             layers=[2, 2, 2, 2],
                                             se=args.se,
                                             in_channels=1,
                                             low_rate=1,
                                             alpha=args.alpha,
                                             t2s_mul=args.t2s_mul,
                                             beta=args.beta,
                                             base_channel=args.base_channel)

        self.high_rate_branch = HighRateBranch(block=BasicBlock,
                                               layers=[2, 2, 2, 2],
                                               se=args.se,
                                               in_channels=1,
                                               low_rate=0,
                                               alpha=args.alpha,
                                               t2s_mul=args.t2s_mul,
                                               beta=args.beta,
                                               base_channel=args.base_channel)

    def forward(self, x, y):
        b = x.size(0)
        y, laterals = self.high_rate_branch(y)
        x = self.low_rate_branch(x, laterals)

        x = x.view(b, -1, 8*self.args.base_channel+8*self.args.base_channel//self.args.alpha*self.args.t2s_mul)
        return x


class MSTP(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(MSTP, self).__init__()
        self.args = args
        self.mbranch = MultiBranchNet(args)
        in_dim = 8 * args.base_channel + 8 * args.base_channel // self.args.alpha * self.args.t2s_mul
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.v_cls = nn.Linear(1024*2, self.args.n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, event_low, event_high):
        self.gru.flatten_parameters()
        if self.training:
            with autocast():
                feat = self.mbranch(event_low, event_high)
                feat = self.dropout(feat)
                feat = feat.float()
        else:
            feat = self.mbranch(event_low, event_high)
            feat = feat.float()

        feat, _ = self.gru(feat)
        logit = self.v_cls(self.dropout(feat)).mean(1)

        return logit

