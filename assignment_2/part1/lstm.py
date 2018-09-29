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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.device = device

        W_gx = torch.randn(num_hidden, input_dim, device=self.device)
        nn.init.xavier_uniform_(W_gx)
        self.W_gx = nn.Parameter(W_gx)

        W_gh = torch.randn(num_hidden, num_hidden, device=self.device)
        nn.init.xavier_uniform_(W_gh)
        self.W_gh = nn.Parameter(W_gh)

        W_ix = torch.randn(num_hidden, input_dim, device=self.device)
        nn.init.xavier_uniform_(W_ix)
        self.W_ix = nn.Parameter(W_ix)

        W_ih = torch.randn(num_hidden, num_hidden, device=self.device)
        nn.init.xavier_uniform_(W_ih)
        self.W_ih = nn.Parameter(W_ih)

        W_fx = torch.randn(num_hidden, input_dim, device=self.device)
        nn.init.xavier_uniform_(W_fx)
        self.W_fx = nn.Parameter(W_fx)

        W_fh = torch.randn(num_hidden, num_hidden, device=self.device)
        nn.init.xavier_uniform_(W_fh)
        self.W_fh = nn.Parameter(W_fh)

        W_ox = torch.randn(num_hidden, input_dim, device=self.device)
        nn.init.xavier_uniform_(W_ox)
        self.W_ox = nn.Parameter(W_ox)

        W_oh = torch.randn(num_hidden, num_hidden, device=self.device)
        nn.init.xavier_uniform_(W_oh)
        self.W_oh = nn.Parameter(W_oh)

        W_ph = torch.randn(num_classes, num_hidden, device=self.device)
        nn.init.xavier_uniform_(W_ph)
        self.W_ph = nn.Parameter(W_ph)

        self.b_g = nn.Parameter(torch.zeros(batch_size, device=self.device))
        self.b_i = nn.Parameter(torch.zeros(batch_size, device=self.device))
        self.b_f = nn.Parameter(torch.zeros(batch_size, device=self.device))
        self.b_o = nn.Parameter(torch.zeros(batch_size, device=self.device))
        self.b_p = nn.Parameter(torch.zeros(batch_size, device=self.device))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        h_t = torch.zeros(self.num_hidden, self.batch_size, device=self.device)
        c_t = torch.zeros(self.num_hidden, self.batch_size, device=self.device)

        x = x.to(self.device)

        for i in range(self.seq_length):
            x_t = x[:, i].view(1, -1)
            g_t = self.tanh(torch.mm(self.W_gx, x_t) + torch.mm(self.W_gh, h_t) + self.b_g)
            i_t = self.sigmoid(torch.mm(self.W_ix, x_t) + torch.mm(self.W_ih, h_t) + self.b_i)
            f_t = self.sigmoid(torch.mm(self.W_fx, x_t) + torch.mm(self.W_fh, h_t) + self.b_f)
            o_t = self.sigmoid(torch.mm(self.W_ox, x_t) + torch.mm(self.W_oh, h_t) + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t

        p_t = torch.mm(self.W_ph, h_t) + self.b_p
        y_t = self.softmax(p_t)
        return torch.t(y_t)