import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from pathlib import Path
from data_collector.market_collector import MarketDataCollector
from data_collector.news_collector import NewsCollector
from features.indicators import TechnicalIndicators
from features.sentiment import LlamaSentimentAnalyzer
from environment.trading_env import StockTradingEnv
from agents.train_offline import OfflineTrainer
import logging
import sys

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("[BOT] COMPLETE FREE RL TRADING BOT WITH LOCAL LLAMA 3.2")
    print("="*70)
    
    # STEP 1: Download market data (FREE via yfinance)
    print("\n[DATA] STEP 1: Downloading market data from Yahoo Finance (FREE)...")
    collector = MarketDataCollector()
    try:
        market_data = collector.download_historical_data(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2024-10-30'
        )
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        sys.exit(1)
        
    # STEP 2: Fetch financial news (FREE via RSS)
    print("\n[NEWS] STEP 2: Fetching financial news from RSS feeds (FREE)...")
    news_collector = NewsCollector()
    news = news_collector.fetch_news(hours=24)
    
    # STEP 3: Analyze sentiment with LOCAL Llama 3.2 (FREE, NO API CALLS)
    print("\n[AI] STEP 3: Analyzing sentiment with LOCAL Llama 3.2...")
    analyzer = LlamaSentimentAnalyzer()
    
    if news:
        print(f"   Analyzing {min(3, len(news))} headlines...")
        for item in news[:3]:
            result = analyzer.analyze_sentiment(item['title'])
            print(f"   [*] {item['title'][:50]}...")
            print(f"      Sentiment: {result['sentiment'].upper()}, Score: {result['score']:.2f}")
    
    # STEP 4: Calculate technical indicators (FREE)
    print("\n[TECH] STEP 4: Calculating technical indicators (FREE)...")
    indicators = TechnicalIndicators.calculate_indicators(market_data, ticker='AAPL')
    
    # STEP 5: Create trading environment (FREE - Gymnasium)
    print("\n[ENV] STEP 5: Creating Gymnasium trading environment (FREE)...")
    env = StockTradingEnv(indicators, initial_balance=100000)
    obs, _ = env.reset()
    print(f"   [OK] Environment created")
    print(f"   State shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    # STEP 6: Train RL agent (FREE - Stable-Baselines3)
    print("\n[TRAIN] STEP 6: Training RL agent on historical data (FREE)...")
    print("   (This may take a few minutes...)")
    
    split_idx = int(len(indicators) * 0.8)
    train_data = indicators.iloc[:split_idx]
    val_data = indicators.iloc[split_idx:]
    
    env_train = StockTradingEnv(train_data, initial_balance=100000)
    env_val = StockTradingEnv(val_data, initial_balance=100000)
    
    trainer = OfflineTrainer()
    model = trainer.train_agent(
        env_train, env_val,
        agent_type='PPO',
        total_timesteps=10000  # Small for demo
    )
    
    print("\n" + "="*70)
    print("[SUCCESS] COMPLETE!")
    print("="*70)
    print("\nYou've successfully created:")
    print("  [OK] Market data collection (FREE - yfinance)")
    print("  [OK] News scraping (FREE - RSS feeds)")
    print("  [OK] Sentiment analysis (FREE - Local Llama 3.2)")
    print("  [OK] Technical indicators (FREE - pandas-ta)")
    print("  [OK] RL environment (FREE - Gymnasium)")
    print("  [OK] Trained RL agent (FREE - Stable-Baselines3)")
    print("\nAll components are completely FREE and open-source!")
    print("\nNext steps:")
    print("  1. Run continuous learning on paper trading")
    print("  2. Backtest your strategies")
    print("  3. Deploy to cloud for 24/7 trading")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()