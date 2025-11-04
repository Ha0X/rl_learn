import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 128

class Critic(nn.Module):
    """状态价值网络 V(s)"""
    def __init__(self, obs_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.v = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, s: torch.Tensor):
        x = F.relu(self.fc1(s))
        return self.v(x)


class GaussianTanhActor(nn.Module):
    """
    高斯策略 + tanh squash
    输出: 均值 μ、标准差 σ
    sample(): 返回 a_tanh, log_prob
    """
    def __init__(self, obs_dim: int, act_dim: int, log_std_min=-6, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.mu_head = nn.Linear(HIDDEN_SIZE, act_dim)
        self.log_std_head = nn.Linear(HIDDEN_SIZE, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s: torch.Tensor):
        x = F.relu(self.fc1(s))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x),
                              self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std, log_std

    def sample(self, s: torch.Tensor):
        """TODO: 实现高斯采样 + tanh + log_prob 修正"""
        raise NotImplementedError
