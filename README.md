# 📈 QuantTrader: AI-Powered RL Trading Bot

**QuantTrader** is an open-source, privacy-focused trading bot that combines **Reinforcement Learning (RL)** with **Local Large Language Models (LLMs)** for sentiment analysis. It operates entirely on free data sources and local compute, eliminating the need for expensive API subscriptions.

The system uses **Proximal Policy Optimization (PPO)** to train agents on a custom gymnasium environment, enriched with both technical indicators and news sentiment derived from a local **Llama 3.2** or **DeepSeek** model.

---

## 🚀 Key Features

* **Free Market Data**: Historical and real-time OHLCV data via `yfinance`.
* **News & Sentiment**: Aggregates news from major financial RSS feeds (CNBC, MarketWatch, etc.) and analyzes sentiment using **Local LLMs** (no API costs, private).
* **Reinforcement Learning**: Custom `Gymnasium` environment (`StockTradingEnv`) for training agents using **Stable-Baselines3**.
* **Feature Engineering**: Robust pipeline combining technical indicators (RSI, MACD, Bollinger Bands) with AI-derived sentiment scores.
* **Paper Trading**: Built-in module to simulate trading and log portfolio performance.
* **Privacy First**: All AI inference runs locally using `llama.cpp`.

---

## 📂 Project Structure

```text
quantTrader/
├── backend/
│   ├── agents/           # RL Agent training logic (PPO, A2C, DDPG)
│   ├── data_collector/   # Market data (yfinance) and News (RSS) fetchers
│   ├── environment/      # Custom Gymnasium StockTradingEnv
│   ├── features/         # Technical indicators & LLM Sentiment analysis
│   ├── logs/             # Model checkpoints and training logs
│   ├── raw_data/         # CSV storage for fetched market/news data
│   ├── config.py         # Configuration for LLM paths and GPU settings
│   ├── main.py           # Main pipeline entry point
│   └── paper_trading.py  # Paper trading execution script
├── requirements.txt      # Project dependencies
├── NEXT_STEPS.md         # Development roadmap and bug tracking
└── PROJECT_ASSESSMENT.md # Current project status report
