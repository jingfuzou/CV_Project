import torch


class VGG16(torch.nn.Module):
    def __init__(self, n_class=21):
        super(VGG16, self).__init__()
        #con1
        self.conv1_1 = torch.nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1) # same convolution
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/2
        # conv2
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, padding=1) # same convolution
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4
        # conv3
        self.conv3_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/8
        # conv4
        self.conv4_1 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16
        # conv5
        self.conv5_1 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32

        # 全连接层
        self.fc6 = torch.nn.Linear(512*7*7, 4096)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.fc7 = torch.nn.Linear(4096, 4096)
        self.relu7 = torch.nn.ReLU(inplace=True)

    # 使用定义的VGG16， 网络执行的过程
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

