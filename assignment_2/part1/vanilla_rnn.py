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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

        W_hx = torch.randn(num_hidden, input_dim, device=self.device)
        torch.nn.init.xavier_uniform_(W_hx)
        self.W_hx = nn.Parameter(W_hx)

        W_hh = torch.randn(num_hidden, num_hidden, device=self.device)
        torch.nn.init.xavier_uniform_(W_hh)
        self.W_hh = nn.Parameter(W_hh)
        W_ph = torch.randn(num_classes, num_hidden, device=self.device)
        torch.nn.init.xavier_uniform_(W_ph)
        self.W_ph = nn.Parameter(W_ph)

        self.b_h = nn.Parameter(torch.zeros(batch_size, device=self.device))
        self.b_p = nn.Parameter(torch.zeros(batch_size, device=self.device))

        self.tanh = nn.Tanh()


    def forward(self, x):
        h = torch.zeros(self.num_hidden, self.batch_size, device=self.device)
        x = x.to(self.device)

        for i in range(self.seq_length):
            x_t = x[:,i].view(1,-1)
            h = self.tanh(torch.mm(self.W_hx, x_t) + torch.mm(self.W_hh, h) + self.b_h)

        p_t = torch.mm(self.W_ph, h) + self.b_p
        return torch.t(p_t)
