# REINFORCE (Policy Gradient) — Minimal Project

This is a tiny, **runnable** PyTorch project to learn CartPole-v1 with REINFORCE.
It includes a baseline option (value head) to reduce variance, but you can switch it off for pure REINFORCE.

## Quickstart

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt

# train (pure REINFORCE)
python main.py --env CartPole-v1 --total-steps 100000 --lr 3e-3 --gamma 0.99 --no-baseline

# train (REINFORCE + baseline / advantage)
python main.py --env CartPole-v1 --total-steps 150000 --lr 3e-3 --gamma 0.99 --baseline --entropy-coef 0.01

# watch tensorboard (optional)
tensorboard --logdir runs
```

## Files
- `main.py` — CLI entry; sets up env/agent/training loop.
- `src/policy.py` — Policy network (categorical); optional value head for baseline.
- `src/train.py` — Monte Carlo trajectory collection + returns/advantages + updates.
- `src/utils.py` — Helpers (seeding, normalization, logging).
- `requirements.txt` — Dependencies.
