#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# models/discriminator.py
import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac  # Pseudo-discriminators' concatenation dimension
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]  # LeakyReLU and Dropout for stability
            dim = item
        seq += [nn.Linear(dim, 1)]  # Final output layer (single output)
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_
        return gradient_penalty

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0  # Ensure correct input dimensions
        return self.seq(input_.view(-1, self.pacdim))  # Forward pass through the layers

