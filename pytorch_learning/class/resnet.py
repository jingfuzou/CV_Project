# 目的：扩展性
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # 记录经过block之后channel的变化量

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        # downsample: 调整纬度一致之后才能相加
        # norm_layer:batch normalization_layer
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果bn层没有自定义， 就使用标准bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.relu(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # downsample调整x的维度， F(x)+x一致才能相加

        out += identity
        out = self.relu(out)  # 先相加再激活

        return out


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果bn层没有自定义， 就使用标准bn层

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self):
        indentity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            indentity = self.downsample(x)

        out += indentity


class ResNet(nn.Module):
    def __init__(self, name, block, layers, num_class=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.name = name
        self._norm_layer = norm_layer

        self.inplanes = 64

        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[0])
        self.layer3 = self._make_layer(block, 256, layers[0])
        self.layer4 = self._make_layer(block, 512, layers[0])
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))  # （1，1）等于GAP 全局平均池化
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时spatialhechannel两个方向
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # Sequential 层组合blocks, 形成stage.如果layers=[2,3,4]， 那么* layers=?

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet('resnet18', BasicBlock, [2, 2, 2, 2])


def resnet50():
    return ResNet('resnet50', BottleBlock, [3, 4, 6, 3])


def resnet152():
    return ResNet('resnet152', BottleBlock, [3, 8, 36, 3])
