################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
from extra.laplotter import LossAccPlotter
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

MODEL_FOLDER = 'models/'
IMAGES_FOLDER = 'images_opt/'

def get_accuracy(predictions, targets):
    accuracy = float(torch.sum(predictions.argmax(dim=1) == targets)) / predictions.shape[0]
    return accuracy

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    if not os.path.isdir(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)

    filename = config.model_type + '_nods' + '_length_input=' + str(config.input_length) + '_optimizer=' + config.optimizer + '_lr=' + str(config.learning_rate).replace('.',',')
    print("Training " + config.model_type + " " + str(config.input_length) + " optimizer " + config.optimizer + ' lr ' + str(config.learning_rate))

    f = open(MODEL_FOLDER + filename, 'w')
    plotter = LossAccPlotter(config.model_type + ' input length ' + str(config.input_length) + ' optimizer ' + config.optimizer, IMAGES_FOLDER + filename, x_label="Steps", show_regressions=False)

    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                           config.batch_size)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                     config.batch_size)

    #print([print (x.shape) for x in model.parameters()])

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        predictions = model(batch_inputs.to(device))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        loss = criterion(predictions, batch_targets)
        accuracy = get_accuracy(predictions, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 100 == 0:
            info = "Train Step {:04d}/{:04d}: Accuracy = {:.2f}, Loss = {:.3f}".format(step, config.train_steps, accuracy, loss)
            f.write(info + '\n')

            plotter.add_values(step, loss_train=loss.data.numpy(), acc_train=accuracy, redraw=False)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    plotter.redraw(plot=False)
    f.close()
    print('Done training.')


 ################################################################################
 ################################################################################


def input_length_exploration():
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--optimizer', type=str, default="adam", help="optimizer to use")

    config = parser.parse_args()

    for model in ["RNN","LSTM"]:
        for input_length in [15,20,30]:
            for optimizer in ["adam", "rmsprop"]:
                config.model_type = model
                config.input_length = input_length
                config.optimizer = optimizer
                train(config)

if __name__ == "__main__":
    input_length_exploration()
