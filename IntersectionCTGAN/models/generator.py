#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# models/generator.py
import torch
from torch import nn
from models.residual import Residual

class Generator(nn.Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]  # Use the residual block here
            dim += item
        seq.append(nn.Linear(dim, data_dim))  # Final layer for the generator
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)  # Pass input through the sequential layers
        return data

