"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    weight = np.random.normal(0, 0.0001, (in_features, out_features))
    grad_weight = np.zeros((in_features, out_features))
    bias = np.zeros(out_features)

    self.params = {'weight': weight, 'bias': bias}
    self.grads = {'weight': grad_weight, 'bias': bias}


  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module                                                 #
    """
    self.x = x

    self.out = np.dot(x, self.params['weight']) + self.params['bias']

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    """

    self.grads['weight'] = np.dot(self.x.T, dout)
    self.grads['bias'] = np.sum(dout, axis = 0)

    dout = np.dot(dout, self.params['weight'].T)

    return dout

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    self.x = np.maximum(0, x)

    return self.x

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    """

    drelu = np.zeros_like(self.x)
    drelu[self.x > 0 ] = 1

    return dout * drelu

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    b = x.max()
    y = np.exp(x-b)
    x = (y.T / y.sum(axis = 1)).T

    self.x = x

    return x

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    """

    dx = np.zeros(dout.shape, dtype=np.float64)

    for i in range(0, dout.shape[0]):
      delta = self.x[i, :].reshape(-1, 1)
      delta = np.diagflat(delta) - np.dot(delta, delta.T)
      dx[i, :] = np.dot(delta, dout[i, :])

    return dx


class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    """

    out = -np.log(x[np.arange(x.shape[0]), y.argmax(1)]).mean()

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    """

    dx = -(y / x) / y.shape[0]


    return dx
