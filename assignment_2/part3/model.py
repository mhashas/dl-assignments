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

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, embedding_size = 300,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x, batch_size = None):
        if not batch_size:
            batch_size = self.batch_size

        h_0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden, device=self.device)
        c_0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden, device=self.device)

        x = self.embedding(x)
        out = self.lstm(x, (h_0, c_0))

        #return self.linear(out.view(-1, out.size(2)))
        return self.linear(out[:,-1])