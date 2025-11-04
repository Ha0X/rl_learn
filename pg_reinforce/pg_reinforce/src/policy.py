import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, baseline=False):
        super().__init__()
        self.baseline = baseline
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden, act_dim)
        if self.baseline:
            self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.net(obs)
        logits = self.pi_head(x)
        dist = torch.distributions.Categorical(logits=logits)
        if self.baseline:
            v = self.v_head(x).squeeze(-1)
            return dist, v
        return dist, None
