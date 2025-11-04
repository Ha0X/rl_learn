# q_learning_frozenlake.py
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "FrozenLake-v1"
    is_slippery: bool = False     
    gamma: float = 0.99           # 折扣因子
    alpha: float = 0.8            # 学习率
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000 # 线性衰减到 eps_end
    train_episodes: int = 20000
    eval_episodes: int = 100

def make_env(cfg: Config):
    env= gym.make(cfg.env_id, is_slippery=cfg.is_slippery)
    return env

def init_q_table(env):
    nS = env.observation_space.n
    nA = env.action_space.n
    q = np.zeros((nS, nA), dtype=np.float32)
    return q

def epsilon_greedy(q: np.ndarray, s: int, epsilon: float, nA: int) -> int:
    if np.random.rand()<epsilon:
        return np.random.randint(nA)
    else:
        return np.argmax(q[s])

def td_update(q: np.ndarray, s: int, a: int, r: float, s2: int, done: bool, cfg: Config):
    q[s,a] = q[s,a] +cfg.alpha * (r +cfg.gamma * np.max(q[s2])-q[s,a])

# sarsa:
# def td_update(q: np.ndarray, s: int, a: int, r: float, s2: int, a2: int | None, done: bool, cfg: Config):
#     target = r if done else r + cfg.gamma * q[s2, a2]
#     q[s, a] += cfg.alpha * (target - q[s, a])


def run_episode(env, q: np.ndarray, cfg: Config, epsilon: float):
    ep_ret, ep_len = 0, 0
    s, _ = env.reset()
    nA = env.action_space.n
    while True:
        a= epsilon_greedy(q, s, epsilon, nA)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        td_update(q, s, a, r, s2, done, cfg)
        s = s2
        ep_ret += r
        ep_len += 1
        if done:
            return ep_ret, ep_len



def evaluate_policy(env, q: np.ndarray, episodes=100):
    total = 0.0
    nA = env.action_space.n

    for _ in range(episodes):
        s, _ = env.reset()
        ep_ret = 0.0
        done = False

        while not done:
            a = int(np.argmax(q[s])) 
            s, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            done = terminated or truncated

        total += ep_ret

    return total / episodes

def main():
    cfg = Config()
    env = make_env(cfg)
    q = init_q_table(env)

    # 训练
    eps = cfg.eps_start
    rewards = []
    for ep in range(1, cfg.train_episodes + 1):
        # 线性衰减 epsilon
        t = min(ep / cfg.eps_decay_episodes, 1.0)
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t

        ep_ret, ep_len = run_episode(env, q, cfg, eps)
        rewards.append(ep_ret)

        if ep % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"[train] ep={ep:4d} avg_return(100ep)={avg:.3f} eps={eps:.3f}")

    # 评估
    avg_eval = evaluate_policy(env, q, cfg.eval_episodes)
    print(f"[eval] avg_return over {cfg.eval_episodes} episodes: {avg_eval:.3f}")

if __name__ == "__main__":
    main()
