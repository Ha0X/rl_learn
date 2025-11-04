import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from .policy import MLPPolicy
from .utils import compute_returns, normalize

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = MLPPolicy(obs_dim, act_dim, baseline=args.baseline).to(device)
    optim = Adam(policy.parameters(), lr=args.lr)
    writer = SummaryWriter()

    global_step = 0
    ep_return = 0.0
    ep_len = 0
    ep_count = 0

    while global_step < args.total_steps:
        # --- Collect a batch of trajectories (approximately batch-steps) ---
        batch_obs, batch_acts, batch_rews, batch_dones, batch_logp = [], [], [], [], []

        steps_collected = 0
        obs, _ = env.reset()
        done = False
        while steps_collected < args.batch_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist, _ = policy(obs_t)
            act = dist.sample()
            logp = dist.log_prob(act)

            next_obs, r, terminated, truncated, _ = env.step(act.item())
            done = terminated or truncated

            batch_obs.append(obs)
            batch_acts.append(act.item())
            batch_rews.append(r)
            batch_dones.append(done)
            batch_logp.append(logp.detach().cpu().numpy())

            ep_return += r
            ep_len += 1
            global_step += 1
            steps_collected += 1

            obs = next_obs
            if done:
                writer.add_scalar("charts/episode_return", ep_return, global_step)
                writer.add_scalar("charts/episode_length", ep_len, global_step)
                ep_return, ep_len = 0.0, 0
                ep_count += 1
                obs, _ = env.reset()

        # --- Compute returns/advantages (Monte Carlo) ---
        returns = compute_returns(batch_rews, batch_dones, args.gamma)
        returns = np.asarray(returns, dtype=np.float32)

        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int64, device=device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # Optionally normalize returns for stability (pure REINFORCE)
        if not args.baseline:
            ret_tensor = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)

        # --- Compute losses ---
        if args.baseline:
            # Advantage = returns - V(s)
            dist, v_pred = policy(obs_tensor)
            logp = dist.log_prob(act_tensor)
            with torch.no_grad():
                adv = ret_tensor - v_pred
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -(logp * adv).mean()
            value_loss = 0.5 * (v_pred - ret_tensor).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        else:
            # Pure REINFORCE (no baseline)
            dist, _ = policy(obs_tensor)
            logp = dist.log_prob(act_tensor)
            pg_loss = -(logp * ret_tensor).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss - args.entropy_coef * entropy
            value_loss = torch.tensor(0.0, device=device)

        # --- Optimize ---
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        # --- Logging ---
        writer.add_scalar("loss/total", loss.item(), global_step)
        writer.add_scalar("loss/policy", pg_loss.item(), global_step)
        writer.add_scalar("loss/value", value_loss.item() if args.baseline else 0.0, global_step)
        writer.add_scalar("stats/entropy", entropy.item(), global_step)
        writer.add_scalar("stats/episodes", ep_count, global_step)

        if (global_step // args.batch_steps) % args.log_interval == 0:
            print(f"step={global_step}  loss={loss.item():.3f}  pg={pg_loss.item():.3f}  "
                  f"entropy={entropy.item():.3f}  episodes={ep_count}")

    env.close()
    writer.close()
