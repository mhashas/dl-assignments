"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
from torch import nn
from torch.autograd import Variable
from constants import *


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2000,1000,500,500,200,100,100'
LEARNING_RATE_DEFAULT = 5e-3
MAX_STEPS_DEFAULT = 20000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 1000

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  """

  pred = np.argmax(predictions, axis=1)
  lab = np.argmax(targets, axis=1)

  sum = 0
  for i in range(len(pred)):
    if pred[i] == lab[i]:
      sum += 1

  accuracy = sum / len(pred)

  return accuracy


def train():
  """
  Performs training and evaluation of MLP model.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

  x = x.reshape(np.size(x,0), -1)
  n_input = np.size(x, 1)
  net = MLP(n_input, dnn_hidden_units, 10).to(device)
  crossEntropy = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr = FLAGS.learning_rate)

  loss_list = []
  accuracy_list = []
  test_eval_list = []

  for i in range(FLAGS.max_steps):
      x = Variable(torch.from_numpy(x), requires_grad = True).to(device)
      predictions = net(x).to(device)
      numpy_predictions = predictions.cpu().data[:].numpy()

      label_index = np.argmax(y, axis=1)
      label_index = torch.LongTensor(label_index).to(device)

      loss = crossEntropy(predictions, label_index)

      if i % FLAGS.eval_freq == 0:
          current_accuracy = accuracy(numpy_predictions, y)
          current_test_accuracy = test(net)
          current_loss = loss.cpu().data.numpy()

          loss_list.append(current_loss)
          accuracy_list.append(current_accuracy)
          test_eval_list.append(current_test_accuracy)

          print('Training epoch %d out of %d. Loss %.3f, Train accuracy %.3f, Test accuracy %.3f' % (
          i, FLAGS.max_steps, current_loss, current_accuracy, current_test_accuracy))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = x.reshape(np.size(x, 0 ), -1)

  # save model
  torch.save(net, MODEL_DIRECTORY + MLP_PYTORCH_FILE)

def test(net = None):
  np.random.seed(42)

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  net = net if net else torch.load(MODEL_DIRECTORY + MLP_PYTORCH_FILE).to(device)
  x = cifar10['test'].images
  y = cifar10['test'].labels

  x = x.reshape(np.size(x, 0), -1)
  x = Variable(torch.from_numpy(x), requires_grad=True).to(device)

  predictions = net(x)
  numpy_predictions = predictions.cpu().data[:].numpy()

  return accuracy(numpy_predictions, y)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
    # Run the training operation
  if FLAGS.evaluate:
      accuracy = test()
      print('Test accuracy %.3f' % (accuracy))
  else:
      train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--evaluate', type=int, default=0,
                      help='If we should train or evaluate')

  FLAGS, unparsed = parser.parse_known_args()

  main()