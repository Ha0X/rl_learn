import math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import pybullet_envs_gymnasium  

# ========== 超参数 ==========
ENV_ID = "ReacherBulletEnv-v0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR_ACTOR = 2e-4
LR_CRITIC = 5e-4
HIDDEN = 128
ENTROPY_BETA_START = 3e-4
ENTROPY_BETA_END   = 0.0

MAX_EPISODES = 500
for ep in range(1, MAX_EPISODES+1):
    entropy_beta = ENTROPY_BETA_START + (ENTROPY_BETA_END-ENTROPY_BETA_START) * (ep/MAX_EPISODES)

# 可选：设随机种子（方便复现实验）
SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ========== 工具函数 ==========
def scale_action(a_tanh, low, high):
    """把 tanh 后位于 (-1,1) 的动作线性映射到环境动作区间 [low, high] """
    return low + (a_tanh + 1.0) * 0.5 * (high - low)


# ========== 网络 ==========
class Critic(nn.Module):
    """V(s) 估计器：输入状态 -> 输出标量 V(s)"""
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        return self.v(x)


class GaussianTanhActor(nn.Module):
    """
    策略 π(a|s):
      1) 先输出高斯 N(μ, σ^2) 的参数
      2) 在 x ~ N(μ, σ^2) 上采样
      3) a = tanh(x) 压到 (-1,1)
      4) 用变量变换公式，修正 log_prob（减去 log(1 - a^2)）
    """
    def __init__(self, obs_dim, act_dim, log_std_min=-6, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, HIDDEN)
        self.mu_head = nn.Linear(HIDDEN, act_dim)
        self.log_std_head = nn.Linear(HIDDEN, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s):
        x = F.relu(self.fc1(s))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std, log_std

    @torch.no_grad()
    def act(self, s, act_low, act_high):
        """
        仅用于评估/收集动作（不求梯度）：返回 env 动作（映射到[low, high]）
        """
        mu, std, _ = self(s)
        eps = torch.randn_like(mu)
        x = mu + std * eps
        a_tanh = torch.tanh(x).squeeze(0).cpu().numpy().astype(np.float32)
        a_env = scale_action(a_tanh, act_low, act_high).astype(np.float32)
        return a_env

    def sample_with_logp(self, s):
        """
        训练用：采样 a_tanh 和它的 log_prob（带 tanh 修正）
        返回：
          a_tanh: 形状 [B, act_dim]，位于 (-1,1)
          log_prob: 形状 [B, 1]
        """
        mu, std, log_std = self(s)
        eps = torch.randn_like(mu)
        x = mu + std * eps                     # 先在高斯上采样（无界）
        a_tanh = torch.tanh(x)                 # 压到 (-1,1)

        # 高斯对数概率（逐维相加）
        log_prob_gauss = (
            -0.5 * ((x - mu) / (std + 1e-8)).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1, keepdim=True)

        # tanh 变量变换修正：da/dx = 1 - tanh(x)^2 = 1 - a^2
        # log π(a) = log π(x) - ∑ log(1 - a^2)
        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob_gauss - correction
        return a_tanh, log_prob


# ========== 训练主流程（一步 TD 更新，边采样边更新） ==========
def main():
    # render_mode=None 训练更快；想看可视化就改为 "human"
    env = gym.make(ENV_ID, render_mode=None)
    obs_space, act_space = env.observation_space, env.action_space
    assert len(obs_space.shape) == 1 and len(act_space.shape) == 1, "需要一维 Box 状态/动作空间"


    obs_dim, act_dim = obs_space.shape[0], act_space.shape[0]
    act_low = act_space.low.astype(np.float32)
    act_high = act_space.high.astype(np.float32)

    actor = GaussianTanhActor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)

    opt_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    for ep in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset(seed=SEED + ep)  # 可选：每回合不同 seed
        ep_ret, ep_len, done = 0.0, 0, False

        while not done:
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # 1) 从策略采样（tanh）并计算 log_prob；同时准备 env 动作（缩放到 [low, high]）
            a_tanh, logp = actor.sample_with_logp(s_t)
            a_env = scale_action(
                a_tanh.squeeze(0).detach().cpu().numpy().astype(np.float32),
                act_low, act_high
            )

            # 2) 与环境交互
            # 环境交互
            next_obs, r, terminated, truncated, _ = env.step(a_env)

            with torch.no_grad():
                v_s = critic(s_t)  # [1,1]
            if terminated:     # 只有“真正终止”才把 bootstrap 置零
                v_next = torch.zeros_like(v_s)
            else:
                ns_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                v_next = critic(ns_t)

            target = torch.tensor([[r]], dtype=torch.float32, device=DEVICE) + GAMMA * v_next
            adv = target - v_s

            done = bool(terminated or truncated)  # 控制 while 循环退出仍然可以


            # 4) 更新 Critic：拟合 V(s) ≈ target
            v_pred = critic(s_t)
            critic_loss = F.mse_loss(v_pred, target)
            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

            # 5) 更新 Actor：最大化 E[logπ(a|s) * A] + 熵正则（鼓励探索）
            #    注意：-logp 的均值就是熵（忽略常数），这里用 ENTROPY_BETA 控制强度
            actor_loss = -(logp * adv.detach() + entropy_beta * (-logp)).mean()
            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_actor.step()

            # 6) 累计统计
            obs = next_obs
            ep_ret += r
            ep_len += 1

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | return={ep_ret:7.2f} | len={ep_len:4d}")

    env.close()


if __name__ == "__main__":
    main()
