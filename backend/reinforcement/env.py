import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

#dataloader = DataLoader(dataset, shuffle=False)

gym.register(
    id = "QuantTrader-v0",
    entry_point="reinforcement.env:QuantTraderEnv"
)

class QuantTraderEnv(gym.Env):
    metadata = {'render_mode': ["human"]}
    def __init__(self, render_mode=None, window_size = 20):
        super().__init__()

        data_dir = Path("raw_data/2026-03-10/recent_crypto_news.csv")
        self.data = pd.read_csv(data_dir, parse_dates=[0], index_col=0)

        self.num_features = self.data.shape[1]

        #spaces
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.window_size, self.num_features), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.current_step = window_size
        self.position = 0
        self.entry_point = self.data("RET_1")[self.current_step-1]
        self.balace = 10000

    def reset(self):
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        self.current_step+=1
        return super().step(action)
    

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step

        obs = self.data.iloc[start:end].values
        return 