import gymnasium as gym
import gymnasium_robotics
import numpy as np
import multiprocessing as mp

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor


# -------------------------------
# Dense reward wrapper
# -------------------------------
class DenseRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        dist = np.linalg.norm(achieved_goal - desired_goal)
        reward = -dist  # dense reward: 距离越小奖励越高
        return obs, reward, terminated, truncated, info


# -------------------------------
# 环境创建函数
# -------------------------------
def make_env(env_id, rank, use_dense=False, seed=0):
    def _init():
        env = gym.make(env_id)
        if use_dense:
            env = DenseRewardWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------
# 收集 expert 数据
# -------------------------------
def collect_trajectories(model, env, num_episodes=1000, save_path="expert_data.npz"):
    trajectories = []
    success_eps = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        # 去掉 batch 维度 (VecEnv 返回的是 [n_envs, ...])
        if isinstance(obs, dict):
            obs = {k: v[0] for k, v in obs.items()}
        else:
            obs = obs[0]

        done = False
        ep_data = {"obs": [], "actions": [], "next_obs": [], "rewards": [], "dones": []}
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step([action])
            # VecEnv 输出也带 batch，去掉
            if isinstance(next_obs, dict):
                next_obs = {k: v[0] for k, v in next_obs.items()}
            else:
                next_obs = next_obs[0]
            reward = reward[0]
            terminated, truncated = terminated[0], truncated[0]
            done = terminated or truncated

            ep_data["obs"].append(obs)
            ep_data["actions"].append(action)
            ep_data["next_obs"].append(next_obs)
            ep_data["rewards"].append(reward)
            ep_data["dones"].append(done)

            obs = next_obs
            if "is_success" in info[0] and info[0]["is_success"] == 1.0:
                success = True

        if success:
            success_eps += 1
            trajectories.append(ep_data)

    np.savez_compressed(save_path, trajectories=trajectories)
    print(f"Saved {success_eps} successful expert episodes → {save_path}")


# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # Linux/macOS 防止 spawn 卡住

    env_id = "FetchPickAndPlace-v4"
    n_envs = 8

    # 并行训练环境 (用 dense reward 加速)
    train_env = SubprocVecEnv([make_env(env_id, i, use_dense=True) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    # 评估环境 (保持稀疏奖励)
    eval_env = DummyVecEnv([make_env(env_id, 999, use_dense=False)])
    eval_env = VecMonitor(eval_env)

    # SAC + HER
    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=16,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=5e-5,       # 稍微低一点
        batch_size=1024,
        gamma=0.98,
        tau=0.05,
        buffer_size=2_000_000,
        learning_starts=10000,    # ⭐ 增大，避免 buffer 太空
        tensorboard_log="./sac_fetch_tensorboard/",
        device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256])),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_fetch/",
        log_path="./logs_fetch/",
        eval_freq=10000 // n_envs,
        deterministic=True,
        render=False,
    )

    # 训练
    model.learn(total_timesteps=10_000_000, callback=eval_callback)
    model.save("sac_fetch_her_expert")

    # 收集 expert 数据
    collect_trajectories(model, eval_env, num_episodes=1000, save_path="expert_data.npz")
