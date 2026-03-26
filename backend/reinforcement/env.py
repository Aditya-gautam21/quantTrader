import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class QuantTraderEnv(gym.Env):
    metadata = {'render_mode': ["human"]}
    def __init__(self, render_mode=None, window_size = 20):
        super().__init__()

        data_dir = Path("raw_data/2026-03-10/recent_crypto_news.csv")
        self.data = pd.read_csv(data_dir, parse_dates=[0], index_col=0)

        self.window_size = 20

        self.num_features = self.data.shape[1]

        #spaces
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.window_size, self.num_features), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.current_step = window_size
        self.position = 0
        self.balance = 0
        self.unrealised_pnl = 0

        self.price_series = np.exp(self.data["RET_1"].cumsum())
        self.entry_price = self.price_series.iloc[self.current_step - 1]
        

    def step(self, action):
        reward = 0
        done = False

        prev_price = self._get_price(self.current_step-1)
        curr_price = self._get_price(self.current_step)

        price_change = (curr_price - prev_price)/prev_price

        if action == 1:
            position = 1
            self.entry_price = curr_price

        elif action == 2:
            position = -1
            self.entry_price = curr_price

        reward = self.position * price_change

        self.balance *= (1+reward)

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {'balance': self.balance}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.balance = 1.0

        obs = self._get_obs()
        return obs, {}
    
    

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step

        obs = self.data.iloc[start:end].values
        return obs
    
    def _get_price(self, step):
        # You don't have raw price, so use proxy (RET_1 cumulative)
        return np.exp(self.data["RET_1"].iloc[:step + 1].sum())

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.4f}, Position: {self.position}")

gym.register(
    id = "QuantTrader-v0",
    entry_point="reinforcement.env:QuantTraderEnv"
)

if __name__ == "__main__":
    env = gym.make("QuantTrader-v0", render_mode="human")

    obs, info = env.reset()

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        env.render()

        if done:
            break