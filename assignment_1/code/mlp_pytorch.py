"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch import functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    """
    super(MLP, self).__init__()

    self.layers = nn.ModuleList()

    for nodes in n_hidden:
      self.layers.append(nn.Linear(n_inputs, nodes))
      n_inputs = nodes
      self.layers.append(nn.ReLU())

    self.layers.append(nn.Linear(n_inputs, n_classes))
    self.layers.append(nn.Softmax(dim = 1))

    print(self.modules)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    """

    for layer in self.layers:
      x = layer(x)

    return x
