"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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

    self.layers = []
    for nodes in n_hidden:
      self.layers.append(LinearModule(n_inputs, nodes))
      n_inputs = nodes

    self.lastlayer = LinearModule(n_inputs, n_classes)

    self.reLU = ReLUModule()
    self.softMax = SoftMaxModule()

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
      x = layer.forward(x)
      x = self.reLU.forward(x)

    x = self.lastlayer.forward(x)

    out = self.softMax.forward(x)

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    """

    dout = self.softMax.backward(dout)

    dout = self.lastlayer.backward(dout)

    rev_layer = self.layers.reverse()
    for layer in self.layers:
      dout = self.reLU.backward(dout)
      dout = layer.backward(dout)

    return
