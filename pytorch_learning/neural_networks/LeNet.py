# 神经网络
#
# 神经网络可以通过 torch.nn 包来构建。
#
# 现在对于自动梯度(autograd)有一些了解，神经网络是基于自动梯度 (autograd)来定义一些模型。一个 nn.Module 包括层和一个方法 forward(input) 它会返回输出(output)。

"""
一个典型的神经网络训练过程包括以下几点：

1.定义一个包含可训练参数的神经网络

2.迭代整个输入

3.通过神经网络处理输入

4.计算损失(loss)

5.反向传播梯度到神经网络的参数

6.更新网络的参数，典型的用一个简单的更新方法：weight = weight – learning_rate *gradient
"""

# 定义神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 你刚定义了一个前馈函数，然后反向传播函数被自动通过 autograd 定义了。你可以使用任何张量操作在前馈函数上。
#
# 一个模型可训练的参数可以通过调用 net.parameters() 返回

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
# print(params)

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 在继续之前，让我们复习一下所有见过的类。
#
# torch.Tensor – A multi-dimensional array with support for autograd operations like backward(). Also holds the
# gradient w.r.t. the tensor. nn.Module – Neural network module. Convenient way of encapsulating parameters,
# with helpers for moving them to GPU, exporting, loading, etc. nn.Parameter – A kind of Tensor,
# that is automatically registered as a parameter when assigned as an attribute to a Module. autograd.Function –
# Implements forward and backward definitions of an autograd operation. Every Tensor operation, creates at least a
# single Function node, that connects to functions that created a Tensor and encodes its history.


