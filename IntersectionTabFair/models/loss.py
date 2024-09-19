#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn

class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_indices, priv_indices, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_indices = underpriv_indices
        self._priv_indices = priv_indices
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        fairness_loss = 0.0
        for underpriv_index, priv_index in zip(self._underpriv_indices, self._priv_indices):
            underpriv_group = x[:, underpriv_index]
            priv_group = x[:, priv_index]

            underpriv_positive_outcomes = torch.mean(underpriv_group * x[:, self._Y_start_index + self._desire_index])
            priv_positive_outcomes = torch.mean(priv_group * x[:, self._Y_start_index + self._desire_index])

            underpriv_rate = underpriv_positive_outcomes / (underpriv_group.sum() + 1e-8)
            priv_rate = priv_positive_outcomes / (priv_group.sum() + 1e-8)
            abs_disparity = abs(underpriv_rate - priv_rate)
            fairness_loss += abs_disparity

        gen_loss = -torch.mean(crit_fake_pred)
        combined_loss = gen_loss + lamda * fairness_loss

        return combined_loss


