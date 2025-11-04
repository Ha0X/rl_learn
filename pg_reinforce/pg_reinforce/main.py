import argparse
from src.train import train
from src.utils import seed_everything

def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE (Policy Gradient) - minimal project")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-steps", type=int, default=2048, help="collect ~this many steps per update")
    p.add_argument("--baseline", action="store_true", help="use a learned value baseline (advantage)")
    p.add_argument("--no-baseline", dest="baseline", action="store_false")
    p.set_defaults(baseline=False)
    p.add_argument("--entropy-coef", type=float, default=0.0, help="entropy bonus for exploration")
    p.add_argument("--value-coef", type=float, default=0.5, help="critic loss coefficient (if baseline)")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    train(args)
