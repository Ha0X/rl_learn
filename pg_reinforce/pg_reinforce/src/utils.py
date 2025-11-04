import random, os
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_returns(rewards, dones, gamma):
    # rewards, dones: lists of length T (bool dones for episode-ends within the batch)
    # Monte Carlo: propagate from the end to start; reset at episode boundary.
    G = 0.0
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0.0
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def normalize(x, eps=1e-8):
    x = np.array(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)
