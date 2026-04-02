"""
Evaluation & backtesting module for the trained PPO agent.

Usage:
    cd backend
    python rl_agent/evaluate.py
    python rl_agent/evaluate.py --model logs/ppo/ppo_final --data logs/ppo/test_data.csv
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – saves PNG instead of showing window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_agent.env import CryptoTradingEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("PPO-Eval")


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #
def compute_metrics(portfolio_values: list, trades: list, initial_balance: float) -> dict:
    pv   = np.array(portfolio_values, dtype=float)
    rets = np.diff(pv) / (pv[:-1] + 1e-8)

    total_return   = (pv[-1] - pv[0]) / pv[0]
    # annualise assuming 4h candles → 6 candles/day → 2190/year
    periods_per_yr = 2190
    ann_return     = (1 + total_return) ** (periods_per_yr / max(len(pv), 1)) - 1

    volatility     = rets.std() * np.sqrt(periods_per_yr) if len(rets) > 1 else 0.0
    sharpe         = ann_return / (volatility + 1e-8)

    peak           = np.maximum.accumulate(pv)
    drawdown       = (pv - peak) / (peak + 1e-8)
    max_drawdown   = drawdown.min()

    # Sortino
    neg_rets       = rets[rets < 0]
    downside_std   = neg_rets.std() * np.sqrt(periods_per_yr) if len(neg_rets) > 1 else 1e-8
    sortino        = ann_return / downside_std

    # Calmar
    calmar         = ann_return / (abs(max_drawdown) + 1e-8)

    # Trade-level stats
    buys  = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]
    n_trades = min(len(buys), len(sells))

    trade_returns = []
    for b, s in zip(buys[:n_trades], sells[:n_trades]):
        trade_returns.append((s["price"] - b["price"]) / (b["price"] + 1e-8))

    win_rate     = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0.0
    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss   = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = gross_profit / (gross_loss + 1e-8)

    return {
        "total_return":    total_return,
        "ann_return":      ann_return,
        "volatility":      volatility,
        "sharpe_ratio":    sharpe,
        "sortino_ratio":   sortino,
        "calmar_ratio":    calmar,
        "max_drawdown":    max_drawdown,
        "win_rate":        win_rate,
        "profit_factor":   profit_factor,
        "n_trades":        len(trades),
        "final_value":     pv[-1],
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Total Return      : {metrics['total_return']:>+.2%}")
    print(f"  Annualised Return : {metrics['ann_return']:>+.2%}")
    print(f"  Volatility (ann.) : {metrics['volatility']:>.2%}")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>.3f}")
    print(f"  Sortino Ratio     : {metrics['sortino_ratio']:>.3f}")
    print(f"  Calmar Ratio      : {metrics['calmar_ratio']:>.3f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']:>.2%}")
    print(f"  Win Rate          : {metrics['win_rate']:>.2%}")
    print(f"  Profit Factor     : {metrics['profit_factor']:>.3f}")
    print(f"  Total Trades      : {metrics['n_trades']}")
    print(f"  Final Portfolio   : ${metrics['final_value']:>,.2f}")
    print("=" * 55 + "\n")


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #
def plot_results(
    df: pd.DataFrame,
    portfolio_values: list,
    trades: list,
    metrics: dict,
    save_path: str = "logs/ppo/backtest.png",
):
    pv   = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    dd   = (pv - peak) / (peak + 1e-8) * 100

    prices = df["Close"].values[: len(pv)]
    steps  = np.arange(len(pv))

    fig = plt.figure(figsize=(16, 12), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(3, 1, hspace=0.35, figure=fig)

    # ── 1. Price + trade markers ──────────────────────────────────── #
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0d0d0d")
    ax1.plot(steps, prices, color="#4a9eff", linewidth=1.0, label="BTC/USDT Close")

    buy_steps  = [t["step"] for t in trades if t["type"] == "BUY"]
    sell_steps = [t["step"] for t in trades if t["type"] == "SELL"]
    buy_prices  = [df["Close"].iloc[min(s, len(df)-1)] for s in buy_steps]
    sell_prices = [df["Close"].iloc[min(s, len(df)-1)] for s in sell_steps]

    ax1.scatter(buy_steps,  buy_prices,  marker="^", color="#00e676", s=60, zorder=5, label="Buy")
    ax1.scatter(sell_steps, sell_prices, marker="v", color="#ff1744", s=60, zorder=5, label="Sell")
    ax1.set_ylabel("Price (USDT)", color="white")
    ax1.set_title("BTC/USDT Price with Trade Signals", color="white", fontsize=13)
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="white"); ax1.spines[:].set_color("#333")

    # ── 2. Equity curve ───────────────────────────────────────────── #
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0d0d0d")
    ax2.plot(steps, pv, color="#00e676", linewidth=1.5, label="Portfolio Value")
    ax2.axhline(pv[0], color="#888", linestyle="--", linewidth=0.8, label="Initial Balance")
    ax2.fill_between(steps, pv[0], pv, where=(pv >= pv[0]), alpha=0.15, color="#00e676")
    ax2.fill_between(steps, pv[0], pv, where=(pv <  pv[0]), alpha=0.15, color="#ff1744")
    ax2.set_ylabel("Portfolio (USDT)", color="white")
    ax2.set_title(
        f"Equity Curve  |  Return {metrics['total_return']:+.2%}  |  "
        f"Sharpe {metrics['sharpe_ratio']:.2f}  |  MaxDD {metrics['max_drawdown']:.2%}",
        color="white", fontsize=11,
    )
    ax2.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333")

    # ── 3. Drawdown ───────────────────────────────────────────────── #
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#0d0d0d")
    ax3.fill_between(steps, dd, 0, color="#ff1744", alpha=0.5)
    ax3.plot(steps, dd, color="#ff1744", linewidth=0.8)
    ax3.set_ylabel("Drawdown (%)", color="white")
    ax3.set_xlabel("Step", color="white")
    ax3.set_title("Drawdown", color="white", fontsize=11)
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    log.info(f"Plot saved → {save_path}")
    plt.close()


# ------------------------------------------------------------------ #
#  Run backtest                                                        #
# ------------------------------------------------------------------ #
def run_backtest(model_path: str, data_path: str, vecnorm_path: str, save_dir: str):
    # ── load data ─────────────────────────────────────────────────── #
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    log.info(f"Test data: {df.shape}  ({df.index[0]} → {df.index[-1]})")

    # ── build env ─────────────────────────────────────────────────── #
    env = CryptoTradingEnv(df, window_size=60, initial_balance=10_000.0)

    # ── load model ────────────────────────────────────────────────── #
    model = PPO.load(model_path, env=DummyVecEnv([lambda: env]))
    log.info(f"Model loaded from {model_path}")

    # ── rollout ───────────────────────────────────────────────────── #
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))

    # ── metrics & plots ───────────────────────────────────────────── #
    metrics = compute_metrics(env.portfolio_values, env.trades, env.initial_balance)
    print_metrics(metrics)

    plot_path = str(Path(save_dir) / "backtest.png")
    plot_results(df, env.portfolio_values, env.trades, metrics, save_path=plot_path)

    # ── save metrics JSON ─────────────────────────────────────────── #
    import json
    metrics_clean = {k: float(v) for k, v in metrics.items()}
    metrics_path  = str(Path(save_dir) / "backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    return metrics


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    p.add_argument("--model",    type=str, default="logs/ppo/ppo_final")
    p.add_argument("--data",     type=str, default="logs/ppo/test_data.csv")
    p.add_argument("--vecnorm",  type=str, default="logs/ppo/vec_normalize.pkl")
    p.add_argument("--save_dir", type=str, default="logs/ppo")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args.model, args.data, args.vecnorm, args.save_dir)
