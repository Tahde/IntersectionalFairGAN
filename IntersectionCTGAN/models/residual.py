#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# models/residual.py
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Residual, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)  # Batch normalization for better gradient flow
        self.relu = nn.ReLU()

    def forward(self, input_):
        out = self.fc(input_)  # Pass through the fully connected layer
        out = self.bn(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation
        return torch.cat([out, input_], dim=1)  # Concatenate the input and output (residual connection)

