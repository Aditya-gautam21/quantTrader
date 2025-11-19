import stable_baselines3 as sb3
import gymnasium as gym
from environment.trading_env import StockTradingEnv
import csv
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('raw/historical_2023-01-01_to_2024-10-30.csv')
env = StockTradingEnv(data, initial_balance=100000, transaction_cost=0.001, max_position=1.0)

model = sb3.PPO.load("logs/models/pretrained_ppo.zip")

obs = env.reset()
done = False
total_rewards = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_rewards += reward
    print(f"Action: {action}")

with open('paper_trading_log.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Action','Reward'])
    obs, _ = env.reset()
    done = False
    while not done:
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        writer.writerow([action[0], reward])

plt.plot(env.portfolio_values)
plt.title('Portfolio Value over time')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.show()