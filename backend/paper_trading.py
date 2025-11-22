import stable_baselines3 as sb3
import gymnasium as gym
from environment.trading_env import StockTradingEnv
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('raw_data/2025-11-20/AAPL/combined_AAPL.csv')
env = StockTradingEnv(data, initial_balance=100000, transaction_cost=0.001, max_position=1.0)

model = sb3.PPO.load("logs/models/pretrained_ppo.zip", device="cpu")

obs, info = env.reset() 

done = False
total_rewards = 0
action_log = [] 
print("Starting Paper Trading...")

while not done:
    action, _states = model.predict(obs, deterministic=True)
  
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    total_rewards += reward
    print(f"Action: {action}, Reward: {reward:.2f}")
    
    # Store for logging
    action_log.append([action[0], reward])

with open('paper_trading_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Action', 'Reward'])
    writer.writerows(action_log)

print(f"Total Reward: {total_rewards}")

# Plotting the curve
plt.plot(env.portfolio_values)
plt.title('Portfolio Value over time')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.show()