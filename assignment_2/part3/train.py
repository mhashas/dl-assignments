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
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from extra.laplotter import  LossAccPlotter
import random

CHECKPOINTS_FOLDER ='checkpoints/'
ASSETS_FOLDER = 'assets/'
################################################################################

def generate_sentence(model, dataset, config):
    char_list = [torch.tensor([random.choice(list(dataset._ix_to_char.keys()))])]

    for i in range(config.seq_length):
        tensor = torch.unsqueeze(torch.unsqueeze(char_list[-1], 0), 0).float().to(device=config.device)
        if i == 0:
            predictions = model(tensor,1)
        else:
            predictions = model(tensor)
        char_list.append(torch.max(predictions, 1)[1])

    chars = [char.cpu().numpy()[0] for char in char_list]
    generated_sentence = dataset.convert_to_string(chars)

    return generated_sentence

def generate_sentences(config, sentence):
    state = torch.load('checkpoints/{}'.format(config.txt_file.split("/",1)[1].replace('.txt','')))
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length, config.batch_size, config.train_steps)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size).to(device=device)

    model.load_state_dict(state['state_dict'])

    char_list = dataset.convert_to_ix(sentence)
    return_list = [[torch.tensor(char)] for char in char_list]

    for i in range(len(sentence) + 50):
        tensor = [torch.tensor([char_list[i]])]
        tensor = torch.unsqueeze(torch.unsqueeze(tensor[-1], 0), 0).float().to(device=config.device)
        if i==0:
            predictions = model(tensor,1)
        else:
            predictions = model(tensor)

        out = torch.max(predictions, 1)[1]
        char_list.append(out)
        out = out.cpu()
        return_list.append([out])

    chars = [char[0].cpu().numpy() for char in return_list]
    generated_sentence = dataset.convert_to_string(chars)

    return generated_sentence


def get_accuracy(predictions, targets):
    accuracy = float(torch.sum(predictions.argmax(dim=1) == targets)) / predictions.shape[0]
    return accuracy

def train(config):

    if not os.path.isdir(CHECKPOINTS_FOLDER):
        os.mkdir(CHECKPOINTS_FOLDER)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length, config.batch_size, config.train_steps)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size).to(device=device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)

    generated_sentences = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        batch_inputs = torch.unsqueeze(torch.stack(batch_inputs), 2).float().to(device=device)
        batch_targets = torch.cat(batch_targets).to(device=device)

        predictions = model(batch_inputs, config.batch_size)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        loss = criterion(predictions, batch_targets)
        accuracy = get_accuracy(predictions, batch_targets)

        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            sentence = generate_sentence(model, dataset, config)
            generated_sentences.append(sentence)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'checkpoints/{}'.format(config.txt_file.split("/",1)[1].replace('.txt','')))

    filename = config.txt_file.replace('.txt', '') + 'generated_sentences.txt'
    f = open(filename, 'w')
    output_string = '\n'.join(generated_sentences)
    f.write(output_string)



    print('Done training.')


 ################################################################################
 ################################################################################

def document_exploration():

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/us_constitution.txt',help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.97, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=100000  , help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--save_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--evaluate', type=bool, default=True)
    config = parser.parse_args()

    for document in ["assets/poems.txt","assets/linux.txt", "assets/shakespeare.txt"]:
        config.txt_file = document

        if config.evaluate:
            sentence = generate_sentences(config, "I like turtles")
            print(sentence)
        else:
            # Train the model
            train(config)

if __name__ == "__main__":

    document_exploration()