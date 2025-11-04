# ppo_clip_reacher.py —— 纯手写 PPO-Clip + GAE（连续动作 / PyBullet Reacher）
import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import pybullet_envs_gymnasium  # 注册 ReacherBulletEnv

# =================== 超参数 ===================
ENV_ID = "ReacherBulletEnv-v0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA   = 0.99
LAMBDA  = 0.97           # GAE λ
CLIP_EPS= 0.2            # PPO clip
HIDDEN  = 128

LR_ACTOR  = 3e-4
LR_CRITIC = 7e-4
ENT_COEF  = 1e-4         # 熵正则（小一点）
VF_COEF   = 0.5          # 值函数损失权重
MAX_GRAD_NORM = 0.5

BATCH_STEPS   = 2048     # 每次采样的总步数
UPDATE_EPOCHS = 8
MINIBATCH     = 256
MAX_EPISODES  = 2000

SEED = 2024
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# =================== 实用函数 ===================
def scale_action(a_tanh, low, high):
    """(-1,1)->[low,high]；若本身就是[-1,1]，这步等价为原值"""
    return low + (a_tanh + 1.0) * 0.5 * (high - low)

class RunningNorm:
    """轻量级观测标准化（在线估计 mean/var）"""
    def __init__(self, eps=1e-8):
        self.mean=None; self.var_num=None; self.count=eps
    def update(self, x: np.ndarray):
        x = x.astype(np.float32)
        if self.mean is None:
            self.mean = x.copy()
            self.var_num = np.ones_like(x, dtype=np.float32)
            self.count = 1.0
        else:
            self.count += 1.0
            d = x - self.mean
            self.mean += d / self.count
            self.var_num += d * (x - self.mean)
    def norm(self, x: np.ndarray):
        if self.mean is None: return x.astype(np.float32)
        var = self.var_num / max(self.count - 1.0, 1.0)
        std = np.sqrt(var + 1e-6)
        return ((x - self.mean) / std).astype(np.float32)

# =================== 模型 ===================
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.ln  = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim, HIDDEN)
        self.v   = nn.Linear(HIDDEN, 1)
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu')); nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.v.weight, gain=0.01); nn.init.zeros_(self.v.bias)
    def forward(self, s):
        s = self.ln(s)
        x = F.relu(self.fc1(s))
        return self.v(x)

class GaussianTanhActor(nn.Module):
    """π(a|s): x~N(μ,σ^2), a=tanh(x) ∈ (-1,1)"""
    def __init__(self, obs_dim, act_dim, log_std_min=-5.0, log_std_max=-1.0):
        super().__init__()
        self.ln  = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim, HIDDEN)
        self.mu_head      = nn.Linear(HIDDEN, act_dim)
        self.log_std_head = nn.Linear(HIDDEN, act_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        # 初始化：小步动，训练更稳
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu')); nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01); nn.init.zeros_(self.mu_head.bias)
        nn.init.uniform_(self.log_std_head.bias, -1.8, -1.2)
    def forward(self, s):
        s = self.ln(s)
        x = F.relu(self.fc1(s))
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std, log_std
    def sample_with_logp(self, s):
        """训练采样：返回 a_tanh 和修正后的 log_prob（带梯度）"""
        mu, std, log_std = self(s)
        eps = torch.randn_like(mu)
        x = mu + std * eps
        a_tanh = torch.tanh(x)
        logp_gauss = (-0.5 * ((x - mu) / (std + 1e-8)).pow(2) - log_std - 0.5*math.log(2*math.pi)).sum(-1, keepdim=True)
        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)  # tanh 修正
        logp = logp_gauss - correction
        return a_tanh, logp

def log_prob_from_tanh_gaussian(actor: GaussianTanhActor, s, a_tanh):
    """给定 (s,a_tanh) 用当前策略重算 logp（PPO 需要）"""
    mu, std, log_std = actor(s)
    a_tanh = torch.clamp(a_tanh, -0.999999, 0.999999)
    x = torch.atanh(a_tanh)  # 反 tanh
    logp_gauss = (-0.5 * ((x - mu) / (std + 1e-8)).pow(2) - log_std - 0.5*math.log(2*math.pi)).sum(-1, keepdim=True)
    correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)
    return logp_gauss - correction  # [B,1]

# =================== GAE（done-aware） ===================
@torch.no_grad()
def compute_gae_from_deltas(rewards, values, next_values, terminated, gamma=0.99, lam=0.97):
    """
    rewards:      [N] float
    values:       [N] V(s_t)
    next_values:  [N] V(s_{t+1}); 若 terminated[t] 则应为 0
    terminated:   [N] bool（真终止；time-limit 截断不算终止）
    """
    device = rewards.device
    deltas = rewards + gamma * next_values - values  # [N]
    nonterminal = torch.tensor((~terminated).astype(np.float32), device=device)  # True->1, terminated->0
    advantages = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        gae = deltas[t] + gamma * lam * nonterminal[t] * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

# =================== 评估（均值动作） ===================
@torch.no_grad()
def eval_policy(env, actor, obs_norm, act_low, act_high, episodes=5):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            obs_n = obs_norm.norm(obs)
            s = torch.tensor(obs_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mu, std, _ = actor(s)
            a_tanh = torch.tanh(mu)  # 用均值，不加噪声
            a_env = scale_action(a_tanh.squeeze(0).cpu().numpy().astype(np.float32), act_low, act_high)
            obs, r, term, trunc, _ = env.step(a_env)
            done = bool(term or trunc)
            ep_ret += r
        total += ep_ret
    return total / episodes

# =================== 训练 ===================
def main():
    env = gym.make(ENV_ID, render_mode=None)
    obs_space, act_space = env.observation_space, env.action_space
    assert len(obs_space.shape) == 1 and len(act_space.shape) == 1
    obs_dim, act_dim = obs_space.shape[0], act_space.shape[0]
    act_low = act_space.low.astype(np.float32)
    act_high = act_space.high.astype(np.float32)

    actor  = GaussianTanhActor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    opt_actor  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    obs_norm = RunningNorm()

    global_step = 0
    episode = 0
    while episode < MAX_EPISODES:
        # ===== 收集一批 =====
        buf_obs, buf_a_tanh = [], []
        buf_values, buf_next_values = [], []
        buf_rewards, buf_term, buf_trunc = [], [], []
        buf_logp_old = []  # PPO 需要旧 logp

        steps = 0
        ep_stats = []
        while steps < BATCH_STEPS:
            obs, _ = env.reset(seed=SEED + episode + steps)
            ep_ret, ep_len = 0.0, 0
            while True:
                obs_norm.update(obs)
                obs_n = obs_norm.norm(obs)
                s_t = torch.tensor(obs_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # 训练采样（需要 logp 的梯度图）；old_logp 仅存数值（detach）
                a_tanh, logp = actor.sample_with_logp(s_t)
                a_env = scale_action(a_tanh.squeeze(0).detach().cpu().numpy().astype(np.float32), act_low, act_high)
                next_obs, r, terminated, truncated, _ = env.step(a_env)

                # 估值：v_t 和 v_{t+1}（真终止 -> v_{t+1}=0；time-limit 截断 -> 用 V(next)）
                v_t = critic(s_t).squeeze(1)  # [1]
                with torch.no_grad():
                    if terminated:
                        v_next = torch.zeros_like(v_t)
                    else:
                        next_obs_n = obs_norm.norm(next_obs)
                        ns_t = torch.tensor(next_obs_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        v_next = critic(ns_t).squeeze(1)

                # 存
                buf_obs.append(obs_n)
                buf_a_tanh.append(a_tanh.detach().squeeze(0).cpu().numpy().astype(np.float32))
                buf_values.append(v_t.detach().squeeze(0))
                buf_next_values.append(v_next.detach().squeeze(0))
                buf_rewards.append(float(r))
                buf_term.append(bool(terminated))
                buf_trunc.append(bool(truncated))
                buf_logp_old.append(logp.detach().squeeze(0).cpu().numpy().astype(np.float32))

                ep_ret += r; ep_len += 1
                steps += 1; global_step += 1
                obs = next_obs
                if terminated or truncated or steps >= BATCH_STEPS:
                    ep_stats.append((ep_ret, ep_len))
                    break
            episode += 1

        # ===== 打包张量 =====
        obs_t    = torch.tensor(np.asarray(buf_obs), dtype=torch.float32, device=DEVICE)       # [N, obs_dim]
        a_tanh_t = torch.tensor(np.asarray(buf_a_tanh), dtype=torch.float32, device=DEVICE)    # [N, act_dim]
        values_t = torch.stack(buf_values).to(DEVICE).float()                                   # [N]
        next_values_t = torch.stack(buf_next_values).to(DEVICE).float()                         # [N]
        rewards_t = torch.tensor(buf_rewards, dtype=torch.float32, device=DEVICE)               # [N]
        term_np  = np.asarray(buf_term, dtype=np.bool_)
        old_logp_t = torch.tensor(np.asarray(buf_logp_old), dtype=torch.float32, device=DEVICE).unsqueeze(1)  # [N,1]

        # ===== GAE（done-aware） =====
        adv_t, ret_t = compute_gae_from_deltas(rewards_t, values_t, next_values_t, term_np, gamma=GAMMA, lam=LAMBDA)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ===== PPO 多轮小批次更新 =====
        N = obs_t.shape[0]
        idx = np.arange(N)
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(idx)
            for st in range(0, N, MINIBATCH):
                mb = idx[st:st+MINIBATCH]
                mb_obs    = obs_t[mb]
                mb_a_tanh = a_tanh_t[mb]
                mb_adv    = adv_t[mb].unsqueeze(1)
                mb_ret    = ret_t[mb].unsqueeze(1)
                mb_oldlogp= old_logp_t[mb]

                # Critic
                v_pred = critic(mb_obs)
                critic_loss = VF_COEF * F.mse_loss(v_pred, mb_ret.detach())
                opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                opt_critic.step()

                # Actor（PPO-Clip）
                new_logp = log_prob_from_tanh_gaussian(actor, mb_obs, mb_a_tanh)  # [B,1]
                ratio = torch.exp(new_logp - mb_oldlogp)                          # importance ratio
                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1.0-CLIP_EPS, 1.0+CLIP_EPS) * mb_adv.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = (-new_logp).mean()  # 粗略熵（近似）

                actor_loss = policy_loss + ENT_COEF * entropy
                opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                opt_actor.step()

        # ===== 打印 & 评估 =====
        avg_ret = float(np.mean([r for r, _ in ep_stats])) if ep_stats else 0.0
        avg_len = float(np.mean([l for _, l in ep_stats])) if ep_stats else 0.0
        print(f"[ep {episode:4d} | steps {global_step:6d}] avg_return={avg_ret:7.2f} | avg_len={avg_len:5.1f}")

        if episode % 100 == 0:
            eval_ret = eval_policy(env, actor, obs_norm, act_low, act_high, episodes=5)
            print(f"[eval deterministic] avg_return={eval_ret:.2f}")

    env.close()

if __name__ == "__main__":
    main()
