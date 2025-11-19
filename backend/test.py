import pandas as pd
from environment.trading_env import StockTradingEnv

# Minimal test data, or load your CSV
data = pd.DataFrame({
    'PRICE': [100, 101, 102],
    'RSI_14': [0.5, 0.6, 0.55],
    'MACD': [0.1, 0.12, 0.11],
    'NEWS_SENTIMENT': [0.4, 0.5, 0.45]
})

env = StockTradingEnv(data)
obs, _ = env.reset()
print("Initial Observation:", obs)
