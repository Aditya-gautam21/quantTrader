import stable_baselines3 as sb3
import gymnasium as gym
from environment.trading_env import StockTradingEnv
import csv

env = StockTradingEnv()

model = sb3.PPO.load("logs/models/pretrained_ppo.zip")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}")

with open('paper_trading_log.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Action','Reward'])
    obs = env.reset()
    done = False
    while not done:
        action, states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        writer.writerow([action[0], reward])

print("ran succesfully")