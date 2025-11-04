# 0) 导入与超参数
import random
import math
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 100_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_EVERY = 1000     # 每隔多少步把在线网络参数拷贝到目标网络
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30_000       # epsilon 从 1.0 线性衰减到 0.05 所需步数
MAX_STEPS = 200_000            # 训练总步数
EVAL_EVERY = 10_000

# 1) 网络结构（Q 网络：输入 state，输出所有动作的 Q 值）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# 2) 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.int64,   device=DEVICE).unsqueeze(-1),
            torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(s2, dtype=torch.float32,device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
        )
    def __len__(self):
        return len(self.buf)

# 3) epsilon-贪心动作选择
def select_action(q_net, state, step, action_dim):
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)
    if random.random() < eps:
        return random.randrange(action_dim), eps
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = q_net(s)
        action = int(q_values.argmax(dim=1).item())
        return action, eps

# 4) 计算目标值 y
@torch.no_grad()
def compute_target_y(target_q_net, rewards, next_states, dones):
    next_q_max = target_q_net(next_states).max(dim=1, keepdim=True)[0]
    y = rewards + GAMMA * (1.0 - dones) * next_q_max
    return y

# 5) 计算损失函数
def compute_loss(q_net, states, actions, target_y):
    q_pred = q_net(states).gather(dim=1, index=actions)
    loss = F.smooth_l1_loss(q_pred, target_y)
    return loss

# 6) 训练主循环
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_q_net = QNetwork(state_dim, action_dim).to(DEVICE)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    replay = ReplayBuffer(REPLAY_SIZE)

    # 预填充回放池
    s, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        a = env.action_space.sample()
        s2, r, done, truncated, _ = env.step(a)
        replay.push(s, a, r, s2, done or truncated)
        s = s2 if not (done or truncated) else env.reset()[0]

    s, _ = env.reset()
    episode_return, step, best_eval = 0.0, 0, -1e9

    while step < MAX_STEPS:
        # 动作选择
        a, eps = select_action(q_net, s, step, action_dim)
        s2, r, done, truncated, _ = env.step(a)
        d = done or truncated
        replay.push(s, a, r, s2, d)

        s = s2
        episode_return += r
        step += 1

        # 采样更新
        states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

        with torch.no_grad():
            target_y = compute_target_y(target_q_net, rewards, next_states, dones)

        loss = compute_loss(q_net, states, actions, target_y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        # 同步目标网络
        if step % TARGET_UPDATE_EVERY == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        if d:
            s, _ = env.reset()
            episode_return = 0.0

        if step % EVAL_EVERY == 0:
            eval_env = gym.make("CartPole-v1")
            eval_ret = evaluate(q_net, eval_env)  # 评估用独立 env
            eval_env.close()
            best_eval = max(best_eval, eval_ret)
            print(f"step={step}  eps={eps:.3f}  eval_return={eval_ret:.1f}  best={best_eval:.1f}  loss={loss.item():.4f}")

    env.close()

# 7) 评估（贪心策略）
@torch.no_grad()
def evaluate(q_net, env, episodes=5):
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        terminated = False
        truncated  = False
        while not (terminated or truncated):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = int(q_net(s_t).argmax(dim=1).item())  # 纯贪心
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
        total += ep_ret
    return total / episodes


# 8) 入口
if __name__ == "__main__":
    train()
