import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 超参数
GAMMA = 0.99
LR = 1e-3
HIDDEN_SIZE = 128
MAX_EPISODES = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 公共层
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        # Actor 分支：输出每个动作的概率
        self.actor = nn.Linear(HIDDEN_SIZE, action_dim)
        # Critic 分支：输出状态价值 V(s)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)  # 动作概率分布
        value = self.critic(x)             # 状态价值 V(s)
        return probs, value


def train():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []

        # 采样一条轨迹
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32, device=DEVICE)
            probs, value = model(state_tensor)

            # 从分布中采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # 保存对数概率、价值
            log_probs.append(dist.log_prob(action))
            values.append(value)

            # 与环境交互
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated

            rewards.append(reward)
            state = next_state

        # 计算返回 Gt（蒙特卡洛）
        # returns = []
        # G = 0
        # for r in reversed(rewards):
        #     G = r + GAMMA * G
        #     returns.insert(0, G)
        # returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # 转为张量
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        # 一步TD目标：target_t = r_t + γ * V(s_{t+1})
        with torch.no_grad():
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)  # [T]
            next_values = torch.cat([values[1:], torch.zeros(1, device=DEVICE)])   # 末步的 V(s_{T+1})=0
            td_targets = rewards_t + GAMMA * next_values                           # [T]

        # Advantage: A_t = target_t - V(s_t)
        advantages = td_targets - values


        # 损失函数：Actor Loss + Critic Loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印结果
        if (episode + 1) % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode+1}: return={total_reward}, loss={loss.item():.3f}")
            
    env.close()


if __name__ == "__main__":
    train()
