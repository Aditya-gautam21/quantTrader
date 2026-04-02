"""
Training pipeline for the PPO crypto trading agent.

Usage:
    cd backend
    python rl_agent/train.py

Walk-forward splits:
    train  : first 70 %
    val    : next  15 %
    test   : last  15 %
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── make sure backend/ is on the path ─────────────────────────────── #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collector.crypto_collector import CryptoDataCollector
from features.indicators import TechnicalIndicators
from features.norm_indicators import NormalizedIndicators
from rl_agent.env import CryptoTradingEnv
from rl_agent.ppo_agent import build_agent, make_vec_env, build_callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("PPO-Train")

# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #
def load_data() -> pd.DataFrame:
    """Fetch live OHLCV, compute indicators, normalise, merge."""
    log.info("Fetching BTC/USDT 4h data …")
    collector = CryptoDataCollector()
    df_ohlcv  = collector.data_collection(symbol="BTC/USDT", timeframe="4h")

    log.info("Computing technical indicators …")
    indicators = TechnicalIndicators.calculate_indicators(df_ohlcv)
    norm       = NormalizedIndicators().normalize_indicators(indicators)

    merged = df_ohlcv.join(norm, how="inner")
    merged.dropna(inplace=True)
    log.info(f"Dataset ready: {merged.shape}  ({merged.index[0]} → {merged.index[-1]})")
    return merged


def split_data(df: pd.DataFrame, train_pct=0.70, val_pct=0.15):
    n      = len(df)
    t_end  = int(n * train_pct)
    v_end  = int(n * (train_pct + val_pct))
    return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def train(args):
    df = load_data()
    df_train, df_val, df_test = split_data(df)
    log.info(f"Train {len(df_train)} | Val {len(df_val)} | Test {len(df_test)} rows")

    # ── save test split for evaluate.py ──────────────────────────── #
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_dir / "test_data.csv")
    log.info(f"Test data saved → {out_dir / 'test_data.csv'}")

    # ── environments ─────────────────────────────────────────────── #
    env_kwargs = dict(
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        drawdown_penalty_coef=args.drawdown_penalty,
        overtrade_penalty_coef=args.overtrade_penalty,
    )

    train_env = make_vec_env(
        lambda: CryptoTradingEnv(df_train, **env_kwargs),
        n_envs=args.n_envs,
        normalize=True,
    )
    val_env = make_vec_env(
        lambda: CryptoTradingEnv(df_val, **env_kwargs),
        n_envs=1,
        normalize=True,
    )

    # ── agent ────────────────────────────────────────────────────── #
    model = build_agent(
        env=train_env,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )

    callbacks = build_callbacks(
        eval_env=val_env,
        checkpoint_dir=str(out_dir / "checkpoints"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
    )

    log.info(f"Training for {args.timesteps:,} timesteps …")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── save final model + normalisation stats ────────────────────── #
    model.save(str(out_dir / "ppo_final"))
    train_env.save(str(out_dir / "vec_normalize.pkl"))
    log.info(f"Model saved → {out_dir / 'ppo_final.zip'}")
    log.info("Training complete. Run:  python rl_agent/evaluate.py")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="Train PPO crypto trading agent")
    p.add_argument("--timesteps",         type=int,   default=500_000)
    p.add_argument("--window_size",       type=int,   default=60)
    p.add_argument("--initial_balance",   type=float, default=10_000.0)
    p.add_argument("--transaction_cost",  type=float, default=0.001)
    p.add_argument("--drawdown_penalty",  type=float, default=0.1)
    p.add_argument("--overtrade_penalty", type=float, default=0.01)
    p.add_argument("--n_envs",            type=int,   default=4)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--n_steps",           type=int,   default=2048)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--n_epochs",          type=int,   default=10)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--clip_range",        type=float, default=0.2)
    p.add_argument("--ent_coef",          type=float, default=0.01)
    p.add_argument("--lstm_hidden",       type=int,   default=128)
    p.add_argument("--lstm_layers",       type=int,   default=2)
    p.add_argument("--eval_freq",         type=int,   default=20_000)
    p.add_argument("--log_dir",           type=str,   default="logs/ppo")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
