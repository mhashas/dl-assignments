"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """

    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(n_channels, 64, 3, stride = 1, padding = 1)
    self.conv1_bn = nn.BatchNorm2d(64)
    self.maxpool1 = nn.MaxPool2d(3, 2, 1)

    self.conv2 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
    self.conv2_bn = nn.BatchNorm2d(128)
    self.maxpool2 = nn.MaxPool2d(3, 2, 1)

    self.conv3_a = nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
    self.conv3_a_bn = nn.BatchNorm2d(256)
    self.conv3_b = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
    self.conv3_b_bn = nn.BatchNorm2d(256)
    self.maxpool3 = nn.MaxPool2d(3, 2, 1)

    self.conv4_a = nn.Conv2d(256, 512, 3, stride = 1, padding = 1)
    self.conv4_a_bn = nn.BatchNorm2d(512)
    self.conv4_b = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
    self.conv4_b_bn = nn.BatchNorm2d(512)
    self.maxpool4 = nn.MaxPool2d(3, 2, 1)

    self.conv5_a = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
    self.conv5_a_bn = nn.BatchNorm2d(512)
    self.conv5_b = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
    self.conv5_b_bn = nn.BatchNorm2d(512)
    self.maxpool5 = nn.MaxPool2d(3, 2, 1)

    self.avgpool = nn.AvgPool2d(1, 1)
    self.linear = nn.Linear(512, n_classes)
    self.softmax = nn.Softmax(dim = 1)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    """
    x = self.maxpool1(F.relu(self.conv1_bn(self.conv1(x))))

    x = self.maxpool2(F.relu(self.conv2_bn(self.conv2(x))))

    x = F.relu(self.conv3_a_bn(self.conv3_a(x)))
    x = F.relu(self.conv3_b_bn(self.conv3_b(x)))
    x = self.maxpool3(x)

    x = F.relu(self.conv4_a_bn(self.conv4_a(x)))
    x = F.relu(self.conv4_b_bn(self.conv4_b(x)))
    x = self.maxpool4(x)

    x = F.relu(self.conv5_a_bn(self.conv5_a(x)))
    x = F.relu(self.conv5_b_bn(self.conv5_b(x)))
    x = self.maxpool5(x)

    x = self.avgpool(x)
    x = self.linear(x.view(-1, 512))

    out = self.softmax(x)

    return out
