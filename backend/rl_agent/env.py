"""
Custom Gymnasium Trading Environment for PPO Agent.

State  : window of N past timesteps × (OHLCV + indicators + portfolio state)
Actions: 0=Hold, 1=Buy/Long, 2=Sell/Short
Reward : Δportfolio_value − transaction_cost − drawdown_penalty − overtrade_penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class CryptoTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,       # 0.1 % taker fee
        max_position_pct: float = 0.95,        # max % of balance per trade
        drawdown_penalty_coef: float = 0.1,
        overtrade_penalty_coef: float = 0.01,
        reward_scaling: float = 100.0,
    ):
        super().__init__()

        # ── data ──────────────────────────────────────────────────────── #
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_cols = [c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
        self.ohlcv_cols   = ["Open", "High", "Low", "Close", "Volume"]
        # all feature columns used in the observation
        self.obs_cols = self.ohlcv_cols + self.feature_cols
        self.n_features = len(self.obs_cols)

        # ── hyper-params ──────────────────────────────────────────────── #
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.overtrade_penalty_coef = overtrade_penalty_coef
        self.reward_scaling = reward_scaling

        # ── spaces ────────────────────────────────────────────────────── #
        # observation: (window_size, n_features + 4 portfolio dims)
        # portfolio dims: [position (0/1), unrealised_pnl, balance_norm, prev_action]
        self.portfolio_dims = 4
        obs_shape = (self.window_size, self.n_features + self.portfolio_dims)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=Hold, 1=Buy, 2=Sell

        # ── state ─────────────────────────────────────────────────────── #
        self._reset_state()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.current_step   = self.window_size
        self.balance        = self.initial_balance
        self.position       = 0          # 0=flat, 1=long
        self.entry_price    = 0.0
        self.shares_held    = 0.0
        self.peak_value     = self.initial_balance
        self.prev_action    = 0
        self.trade_count    = 0
        self.portfolio_values = [self.initial_balance]
        self.trades: list   = []

    def _current_price(self) -> float:
        return float(self.df.loc[self.current_step, "Close"])

    def _portfolio_value(self) -> float:
        price = self._current_price()
        return self.balance + self.shares_held * price

    def _get_obs(self) -> np.ndarray:
        # market window: shape (window_size, n_features)
        window = self.df[self.obs_cols].iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

        # normalise OHLCV columns by the last close so they are scale-free
        last_close = window[-1, self.obs_cols.index("Close")]
        if last_close > 0:
            for col in ("Open", "High", "Low", "Close"):
                idx = self.obs_cols.index(col)
                window[:, idx] /= last_close
            vol_idx = self.obs_cols.index("Volume")
            vol_mean = window[:, vol_idx].mean() + 1e-8
            window[:, vol_idx] /= vol_mean

        # portfolio state (same value broadcast across window)
        pv = self._portfolio_value()
        unrealised_pnl = (
            (self._current_price() - self.entry_price) / (self.entry_price + 1e-8)
            if self.position == 1 else 0.0
        )
        balance_norm = self.balance / self.initial_balance
        portfolio_row = np.array(
            [self.position, unrealised_pnl, balance_norm, self.prev_action],
            dtype=np.float32,
        )
        portfolio_block = np.tile(portfolio_row, (self.window_size, 1))

        obs = np.concatenate([window, portfolio_block], axis=1)
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    # ------------------------------------------------------------------ #
    #  Gym API                                                             #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        price = self._current_price()
        prev_value = self._portfolio_value()

        # ── execute action ────────────────────────────────────────────── #
        trade_cost = 0.0

        if action == 1 and self.position == 0:          # BUY
            invest = self.balance * self.max_position_pct
            trade_cost = invest * self.transaction_cost
            self.shares_held = (invest - trade_cost) / price
            self.balance -= invest
            self.entry_price = price
            self.position = 1
            self.trade_count += 1
            self.trades.append({"step": self.current_step, "type": "BUY", "price": price})

        elif action == 2 and self.position == 1:        # SELL
            proceeds = self.shares_held * price
            trade_cost = proceeds * self.transaction_cost
            self.balance += proceeds - trade_cost
            self.shares_held = 0.0
            self.entry_price = 0.0
            self.position = 0
            self.trade_count += 1
            self.trades.append({"step": self.current_step, "type": "SELL", "price": price})

        # ── advance step ──────────────────────────────────────────────── #
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # ── reward ────────────────────────────────────────────────────── #
        new_value = self._portfolio_value()
        self.portfolio_values.append(new_value)

        # update peak for drawdown
        if new_value > self.peak_value:
            self.peak_value = new_value

        pnl_reward = (new_value - prev_value) / (prev_value + 1e-8)

        drawdown = (self.peak_value - new_value) / (self.peak_value + 1e-8)
        drawdown_penalty = self.drawdown_penalty_coef * max(drawdown - 0.05, 0.0)

        # penalise excessive trading (more than ~1 trade per 10 steps)
        overtrade_penalty = self.overtrade_penalty_coef * max(
            self.trade_count / max(self.current_step - self.window_size, 1) - 0.1, 0.0
        )

        reward = (pnl_reward - drawdown_penalty - overtrade_penalty) * self.reward_scaling
        self.prev_action = action

        info = {
            "portfolio_value": new_value,
            "balance": self.balance,
            "position": self.position,
            "trade_count": self.trade_count,
            "drawdown": drawdown,
        }
        return self._get_obs(), float(reward), done, False, info

    def render(self, mode: str = "human"):
        pv = self._portfolio_value()
        ret = (pv - self.initial_balance) / self.initial_balance * 100
        print(
            f"Step {self.current_step:5d} | "
            f"Price ${self._current_price():>10,.2f} | "
            f"Portfolio ${pv:>10,.2f} | "
            f"Return {ret:+.2f}% | "
            f"Trades {self.trade_count}"
        )
