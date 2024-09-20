#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/model_utils.py
import torch

def save_model(model, path):
    """Save a model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device='cpu'):
    """Load a model from the specified path."""
    #model.load_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    model.to(device)
    print(f"Model loaded from {path}")

def save_optimizer(optimizer, path):
    """Save an optimizer to the specified path."""
    torch.save(optimizer.state_dict(), path)
    print(f"Optimizer saved to {path}")

def load_optimizer(optimizer, path, device='cpu'):
    """Load an optimizer from the specified path."""
    #optimizer.load_state_dict(torch.load(path, map_location=device))
    optimizer.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Optimizer loaded from {path}")

