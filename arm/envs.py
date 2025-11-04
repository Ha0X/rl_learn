# rl_lab/envs.py
import numpy as np
import gymnasium as gym
from typing import Tuple

DEFAULT_ENV_ID = "ReacherBulletEnv-v0"

def make_env(env_id: str = DEFAULT_ENV_ID,
             render_mode: str | None = None
             ) -> Tuple[gym.Env, int, int, np.ndarray, np.ndarray]:
    """
    创建环境并返回:
      env, obs_dim, act_dim, act_low, act_high
    仅支持 1D Box 的观测和动作空间（连续控制）。
    """
    env = gym.make(env_id, render_mode=render_mode)
    obs_space = env.observation_space
    act_space = env.action_space

    # 仅支持一维 Box（向量）观测/动作
    assert hasattr(obs_space, "shape") and len(obs_space.shape) == 1, \
        "This helper assumes 1D Box observations."
    assert hasattr(act_space, "shape") and len(act_space.shape) == 1, \
        "This helper assumes 1D Box actions (continuous)."

    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.shape[0])

    # 保存动作上下界（float32，便于后续与 torch 对齐）
    act_low = act_space.low.astype(np.float32)
    act_high = act_space.high.astype(np.float32)

    return env, obs_dim, act_dim, act_low, act_high


def scale_action(a_tanh: np.ndarray,
                 low: np.ndarray,
                 high: np.ndarray) -> np.ndarray:
    """
    将 [-1, 1] 区间的动作映射到环境动作区间 [low, high]。
    形状需与 low/high 匹配 (act_dim,)。
    """
    # (a_tanh + 1)/2 先映射到 [0,1]，再线性缩放到 [low, high]
    return low + (a_tanh + 1.0) * 0.5 * (high - low)


# -------------------------
# 自测：python -m rl_lab.envs
# -------------------------
if __name__ == "__main__":
    import time
    try:
        # 触发 Bullet 环境注册（若你跑 Reacher，需要先安装并 import）
        import pybullet_envs_gymnasium  # noqa: F401
    except Exception as e:
        print("提示：如果要用 ReacherBulletEnv-v0，请安装并 import pybullet-envs-gymnasium。", e)

    # 1) 创建环境（若 Reacher 不可用，可把 env_id 改成 'Pendulum-v1'）
    env, obs_dim, act_dim, low, high = make_env(DEFAULT_ENV_ID, render_mode="human")
    print(f"[envs.py self-test] obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"low={low}, high={high}")

    # 2) 动作缩放自检
    mid = scale_action(np.zeros(act_dim, dtype=np.float32), low, high)
    assert np.allclose(mid, (low + high) / 2.0), "scale_action midpoint check failed"
    assert np.allclose(scale_action(-np.ones(act_dim, np.float32), low, high), low)
    assert np.allclose(scale_action(+np.ones(act_dim, np.float32), low, high), high)
    print("[envs.py self-test] scale_action OK")

    # 3) 简单可视化 2 秒（随机动作）
    obs, info = env.reset(seed=0)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        action_env = env.action_space.sample().astype(np.float32)
        obs, r, terminated, truncated, info = env.step(action_env)
        if terminated or truncated:
            obs, info = env.reset()
        time.sleep(1.0/60.0)
    env.close()
    print("[envs.py self-test] render OK")
