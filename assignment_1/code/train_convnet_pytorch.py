"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import pickle
import torch
from constants import *
from torch import nn
from torch import functional as F
from torch.autograd import Variable

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
TEST_BATCH_SIZE_DEFAULT = 100
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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

  pred = np.argmax(predictions, axis = 1)
  lab = np.argmax(targets, axis = 1)

  sum = 0
  for i in range(len(pred)):
      if pred[i] == lab[i]:
          sum += 1

  accuracy = sum / len(pred)

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  n_channels = np.size(x, 1)
  net = ConvNet(n_channels, 10).to(device)
  crossEntropy = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

  loss_list = []
  accuracy_list = []
  test_eval_list = []

  for i in range(FLAGS.max_steps):
      x = Variable(torch.from_numpy(x), requires_grad = True).to(device)
      predictions = net(x).to(device)
      numpy_predictions = predictions.cpu().data[:].numpy()

      label_index = torch.LongTensor(np.argmax(y, axis = 1)).to(device)
      loss = crossEntropy(predictions, label_index)

      if i % FLAGS.eval_freq == 0:
          current_accuracy = accuracy(numpy_predictions, y)
          current_test_accuracy = test(net)
          current_loss = loss.cpu().data.numpy()

          loss_list.append(current_loss)
          accuracy_list.append(current_accuracy)
          test_eval_list.append(current_test_accuracy)

          print ('Training epoch %d out of %d. Loss %.3f, Train accuracy %.3f, Test accuracy %.3f' % (i, FLAGS.max_steps, current_loss, current_accuracy, current_test_accuracy))


      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      x, y = cifar10['train'].next_batch(FLAGS.batch_size)

  # save model
  torch.save(net, MODEL_DIRECTORY + CNN_PYTORCH_FILE)
  test_accuracy = test(net)
  print('Test accuracy %.3f' % (test_accuracy))

def test(net = None):
    np.random.seed(42)

    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net if net else torch.load(MODEL_DIRECTORY + CNN_PYTORCH_FILE).to(device)

    x, y = cifar10['test'].next_batch(FLAGS.test_batch_size)
    accuracy_list = []

    number_of_examples = len(cifar10['test'].labels)
    counter = 0

    while counter <= number_of_examples:
        x = Variable(torch.from_numpy(x), requires_grad=False).to(device)
        predictions = net(x).to(device)
        numpy_predictions = predictions.cpu().data[:].numpy()
        accuracy_list.append(accuracy(numpy_predictions, y))
        x, y = cifar10['test'].next_batch(FLAGS.batch_size)
        counter += FLAGS.test_batch_size

    return np.mean(accuracy_list)

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
  if FLAGS.evaluate:
    accuracy = test()
    print('Test accuracy %.3f' % (accuracy))
  else:
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--test_batch_size', type = int, default = TEST_BATCH_SIZE_DEFAULT,
                      help='Batch size to run evaluation.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--evaluate', type=int, default=0,
                      help='If we should train or evaluate')
  FLAGS, unparsed = parser.parse_known_args()

  main()