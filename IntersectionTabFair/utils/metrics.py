#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_demographic_parity(data_x, data_y, indices_pairs):
    disparities = []
    for u_index, p_index in indices_pairs:
        underpriv_mask = data_x[:, u_index] == 1
        priv_mask = data_x[:, p_index] == 1

        if np.any(underpriv_mask) and np.any(priv_mask):
            underpriv_outcomes = data_y[underpriv_mask]
            priv_outcomes = data_y[priv_mask]
            underpriv_odds = np.mean(underpriv_outcomes)
            priv_odds = np.mean(priv_outcomes)
            disparity = priv_odds - underpriv_odds
            disparities.append((u_index, p_index, disparity))
        else:
            disparities.append((u_index, p_index, np.nan))  # Append NaN if there's insufficient data for a meaningful calculation

    return disparities

def calculate_subgroup_metrics(data_x, data_y, data_y_pred, subgroups, S_start_index):
    metrics = {}
    
    for i, subgroup in enumerate(subgroups):
        subgroup_index = S_start_index + i
        mask = data_x[:, subgroup_index] == 1
        
        if np.any(mask):
            subgroup_y_true = data_y[mask]
            subgroup_y_pred = data_y_pred[mask]
            accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
            f1 = f1_score(subgroup_y_true, subgroup_y_pred, zero_division=0)
            metrics[subgroup] = {'accuracy': accuracy, 'f1': f1}
        else:
            metrics[subgroup] = {'accuracy': np.nan, 'f1': np.nan}

    return metrics

