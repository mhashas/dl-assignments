# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from extra.laplotter import LossAccPlotter
from part3.dataset import TextDataset
from part3.model import TextGenerationModel
import random

MODEL_FOLDER = 'models/'
IMAGES_FOLDER = 'images/'

def get_accuracy(predictions, targets):
    o = torch.max(predictions, 1)[1].cpu().numpy()
    t = targets.cpu().numpy()
    compared = np.equal(o, t)
    correct = np.sum(compared)
    accuracy = correct / len(compared)

    return accuracy

def generate_sentence(model, dataset, step):
    # Generate some sentences by sampling from the model
    random_ix = torch.tensor([dataset.co])
    ix_list = [random_ix]

    for i in range(config.seq_length):
        tensor = torch.unsqueeze(torch.unsqueeze(ix_list[-1], 0), 0).float().to(device=config.device)
        out = model(tensor, 1)
        o = torch.max(out, 1)[1]
        ix_list.append(o)
    char_ix = [x.cpu().numpy()[0] for x in ix_list]
    gen_sen = dataset.convert_to_string(char_ix)
    with open('generated_sentences.txt', 'a') as file:
        file.write('{}: {}\n'.format(step, gen_sen))
    pass

def train(config):
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    if not os.path.isdir(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)

    filename = config.model_type + '_length_input=' + str(config.input_length) + '_optimizer=' + config.optimizer
    print("Training " + config.model_type + " " + str(config.input_length) + " optimizer " + config.optimizer)
    f = open(MODEL_FOLDER + filename, 'w')

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    plotter = LossAccPlotter(config.model_type + ' input length ' + str(config.input_length) + ' optimizer ' + config.optimizer, IMAGES_FOLDER + filename, x_label="Steps", show_regressions=False)


    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, device=device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        model.zero_grad()

        inputs = torch.unsqueeze(torch.stack(batch_inputs), 2).float().to(device)
        targets = torch.cat(batch_targets).to(device)

        predictions = model(inputs.to(device))

        loss = criterion(predictions, targets)
        accuracy = get_accuracy(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 100:
            info = "Train Step {:04d}/{:04d}: Accuracy = {:.2f}, Loss = {:.3f}".format(step, config.train_steps,
                                                                                   accuracy, loss)
            f.write(info + '\n')
            plotter.add_values(step, loss_train=loss.data.numpy(), acc_train=accuracy, redraw=False)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.sample_every:
           generate_sentence(model, dataset, step)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    plotter.redraw(plot=False)
    f.close()
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
