import gymnasium as gym
from gymnasium import spaces

env = gym.make("QuantTrader", render_mode="human")

observation, info = env.reset()