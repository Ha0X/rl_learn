import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os

def make_env(rank, seed=0):
    def _init():
        env = gym.make("FetchPickAndPlace-v4")
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    print(">>> Using FetchPickAndPlace-v4 <<<")

    num_cpu = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy="future",
            online_sampling=True,
            max_episode_length=50,
        ),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        verbose=1,
        tensorboard_log="./sac_her_fetch_tensorboard/"
    )

    # 总 timesteps
    TOTAL_TIMESTEPS = 5_000_000
    # 每次保存和评估间隔
    CHECKPOINT_INTERVAL = 500_000
    os.makedirs("checkpoints", exist_ok=True)

    timesteps = 0
    while timesteps < TOTAL_TIMESTEPS:
        # 训练 CHECKPOINT_INTERVAL 步
        model.learn(total_timesteps=CHECKPOINT_INTERVAL, reset_num_timesteps=False, log_interval=10)
        timesteps += CHECKPOINT_INTERVAL

        # 保存模型
        model_path = f"checkpoints/sac_her_fetch_{timesteps//1000}k"
        model.save(model_path)
        print(f"✅ Saved checkpoint: {model_path}.zip")

        # 用 DummyVecEnv 创建渲染环境评估
        eval_env = DummyVecEnv([lambda: gym.make("FetchPickAndPlace-v4")])
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False, deterministic=True)
        print(f"📊 Evaluation at {timesteps} timesteps: mean_reward={mean_reward:.2f}, std={std_reward:.2f}")
        eval_env.close()

    print("🎉 Training finished! Final model saved.")
