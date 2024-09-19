#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
def print2file(buf, outFile):
    with open(outFile, 'a') as outfd:
        outfd.write(buf + '\n')

def log_subgroup_metrics(subgroup_metrics, source, outFile):
    for subgroup, metrics in subgroup_metrics.items():
        accuracy = metrics['accuracy']
        f1 = metrics['f1']
        buf = f'Subgroup ({source}): {subgroup}, Accuracy: {accuracy}, F1 Score: {f1}'
        print(buf)
        print2file(buf, outFile)


def log_dp_metrics(dp_metrics, source, subgroups, S_start_index, outFile):
    for u_index, p_index, disparity in dp_metrics:
        underpriv_subgroup = subgroups[u_index - S_start_index]
        priv_subgroup = subgroups[p_index - S_start_index]
        buf = f'Subgroup ({source}): {priv_subgroup}-{underpriv_subgroup}, DP: {disparity}'
        print(buf)
        print2file(buf, outFile)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


