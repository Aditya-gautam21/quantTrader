"""
Module 3: Custom Trading Environment for Gymnasium (FREE & Open-Source)
This is where the "world" is defined for the RL agent to learn
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    Custom OpenAI Gymnasium environment for stock trading
    
    State: [price, RSI, MACD, sentiment, portfolio_value, position]
    Action: Continuous [-1, 1] where -1=sell all, 0=hold, 1=buy all
    Reward: Daily return + Sharpe bonus
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 data_df,
                 initial_balance=100000,
                 transaction_cost=0.001,
                 max_position=1.0):
        """
        Initialize trading environment
        
        Args:
            data_df: DataFrame with normalized features
            initial_balance: Starting cash in dollars
            transaction_cost: Cost per trade (0.1% = 0.001)
            max_position: Maximum position size (1.0 = 100% of portfolio)
        """
        self.data = data_df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Current state
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_values = [initial_balance]
        self.actions_taken = []
        
        # Gym API requirements
        # State: [price, RSI, MACD, sentiment, portfolio_value, shares]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Action: continuous value [-1, 1]
        # -1 = sell all, 0 = hold, 1 = buy max
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get normalized state vector"""
        row = self.data.iloc[self.current_step]
        
        # Extract features (they should already be normalized 0-1)
        price = row.get('PRICE', 0.5) if 'PRICE' in row.index else 0.5
        rsi = row.get('RSI_14', 0.5) if 'RSI_14' in row.index else 0.5
        macd = row.get('MACD', 0.5) if 'MACD' in row.index else 0.5
        sentiment = row.get('NEWS_SENTIMENT', 0.5) if 'NEWS_SENTIMENT' in row.index else 0.5
        portfolio_value = self.get_portfolio_value() / (self.initial_balance * 2)
        shares_ratio = self.shares / max(1, self.initial_balance // max(price, 1) if price > 0 else 1)
        
        # Handle NaN values
        obs = np.array([price, rsi, macd, sentiment, portfolio_value, shares_ratio], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, 0.0, 1.0)
        
        return obs
    
    def step(self, action):
        """Execute one step of trading"""
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True
            truncated = True
        else:
            done = False
            truncated = False
        
        # Get current price
        current_row = self.data.iloc[self.current_step]
        current_price = current_row.get('PRICE', 100) if 'PRICE' in current_row.index else 100
        
        if current_price <= 0:
            current_price = 100
        
        # Execute action: Convert [-1, 1] to shares to buy/sell
        action_value = float(action[0])
        
        # Calculate target position
        target_cash = self.max_position * self.balance  # Max cash to use
        target_shares = int((action_value + 1.0) / 2.0 * target_cash / current_price)
        
        # Calculate trade
        shares_to_trade = target_shares - self.shares
        
        if shares_to_trade != 0:
            if shares_to_trade > 0:  # BUY
                cost = shares_to_trade * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares += shares_to_trade
                    self.actions_taken.append(('BUY', self.current_step, current_price))
            else:  # SELL
                revenue = -shares_to_trade * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares -= shares_to_trade
                self.actions_taken.append(('SELL', self.current_step, current_price))
        
        # Calculate portfolio value and reward
        portfolio_value = self.get_portfolio_value()
        self.portfolio_values.append(portfolio_value)
        
        # Reward: daily return %
        prev_value = self.portfolio_values[-2]
        daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # Add Sharpe bonus if holding multiple days
        if len(self.portfolio_values) > 30:
            returns = np.array(self.portfolio_values[-30:])
            returns = np.diff(returns) / returns[:-1]
            if len(returns) > 0:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
                reward = daily_return + 0.001 * sharpe
            else:
                reward = daily_return
        else:
            reward = daily_return
        
        return self._get_observation(), float(reward), done, truncated, {}
    
    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        current_row = self.data.iloc[self.current_step]
        current_price = current_row.get('PRICE', 100) if 'PRICE' in current_row.index else 100
        return self.balance + self.shares * current_price

if __name__ == "__main__":
    print("🎮 Testing Gymnasium Trading Environment\n")
    
    n_steps = 100
    dummy_data = pd.DataFrame({
        'PRICE': np.linspace(100, 105, n_steps),
        'RSI_14': np.random.rand(n_steps),
        'MACD': np.random.rand(n_steps),
        'NEWS_SENTIMENT': np.random.rand(n_steps)
    })
    
    # Create environment
    env = StockTradingEnv(dummy_data, initial_balance=10000)
    
    print("Testing random trading for 10 steps...\n")
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample() 
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: Action={action[0]:.2f}, Reward={reward:.6f}, Portfolio=${env.get_portfolio_value():.2f}")
        
        if done or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward:.6f}")
    print(f"Final Portfolio: ${env.get_portfolio_value():.2f}")