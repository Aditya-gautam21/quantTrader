# 📈 QuantTrader: AI-Powered RL Trading Bot

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

An advanced algorithmic trading system that combines **Deep Reinforcement Learning** (PPO) with **Large Language Model sentiment analysis** to make intelligent trading decisions in real-time. Built with Stable-Baselines3 and custom Gymnasium environments.

---

## 🚀 Key Features

### 🤖 Reinforcement Learning
- **PPO Algorithm**: Proximal Policy Optimization for stable policy learning[2][8]
- **Custom Gym Environment**: Purpose-built `StockTradingEnv` with realistic market simulation[11]
- **Multi-Stock Support**: Trade multiple assets simultaneously with portfolio management
- **Risk-Aware Training**: Configurable risk aversion parameters for safer trading strategies[2]

### 📊 Advanced Market Analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **LLM-Powered Sentiment**: Real-time news sentiment analysis using local LLMs[4][7]
- **Multi-Timeframe Data**: Supports various granularities from minute-level to daily data
- **Feature Engineering**: Automated technical indicator calculation and normalization

### 💹 Trading Capabilities
- **Paper Trading Mode**: Test strategies risk-free with simulated execution
- **Real-Time Decision Making**: Live market data integration for production trading
- **Transaction Cost Modeling**: Realistic fee simulation for accurate backtesting[11]
- **Position Management**: Intelligent buy/sell/hold actions with position sizing

### 🔧 Production Ready
- **Model Checkpointing**: Save and resume training with automatic best-model selection
- **Comprehensive Logging**: TensorBoard integration for training visualization
- **Configuration Management**: Centralized config for easy hyperparameter tuning
- **GPU Acceleration**: CUDA support for faster training and inference

---

## 📂 Project Structure

```
quantTrader/
├── agent/                # RL agent implementation (PPO)
│   └── ppo_agent.py     # Training and evaluation logic
├── environment/         # Custom Gymnasium StockTradingEnv
│   └── stock_env.py     # Market simulation environment
├── features/            # Feature engineering
│   ├── technical.py     # Technical indicators (RSI, MACD, etc.)
│   └── sentiment.py     # LLM-based sentiment analysis
├── logs/                # Model checkpoints and training logs
│   └── tensorboard/     # TensorBoard visualization data
├── raw_data/            # CSV storage for market/news data
│   ├── stocks/          # Historical price data
│   └── news/            # News articles for sentiment
├── config.py            # Configuration (LLM paths, GPU settings)
├── main.py              # Main pipeline entry point
├── paper_trading.py     # Paper trading execution script
├── requirements.txt     # Project dependencies
├── NEXT_STEPS.md        # Development roadmap
└── PROJECT_ASSESSMENT.md # Current status report
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for LLM inference)
- 8GB+ RAM recommended

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aditya-gautam21/quantTrader.git
   cd quantTrader
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure LLM settings** (Optional)
   
   Edit `config.py` to set your local LLM path and GPU preferences:
   ```python
   LLM_MODEL_PATH = "/path/to/your/gguf/model"
   USE_GPU = True
   GPU_LAYERS = 32  # Adjust based on your GPU VRAM
   ```

---

## 🚦 Quick Start

### Training a New Agent

```bash
# Train PPO agent on default stocks
python main.py --mode train --tickers AAPL MSFT GOOGL --episodes 1000

# Train with custom parameters
python main.py --mode train \
    --tickers AAPL MSFT \
    --episodes 2000 \
    --learning_rate 0.0003 \
    --batch_size 64
```

### Paper Trading

```bash
# Run paper trading with trained model
python paper_trading.py --model logs/best_model.zip --tickers AAPL MSFT

# Paper trade with real-time data
python paper_trading.py --model logs/best_model.zip --realtime
```

### Backtesting

```bash
# Evaluate model on historical data
python main.py --mode backtest \
    --model logs/best_model.zip \
    --start_date 2023-01-01 \
    --end_date 2024-01-01
```

---

## 📊 Performance Metrics

The system tracks multiple performance indicators[2][8]:

- **Cumulative Return**: Total profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Return**: Mean return per trade
- **Volatility**: Standard deviation of returns

---

## 🧠 Technical Details

### Reinforcement Learning Architecture

- **Algorithm**: Proximal Policy Optimization (PPO)[2][8]
- **Policy Network**: Multi-Layer Perceptron (MLP)
- **Action Space**: Discrete(3) - [Buy, Sell, Hold]
- **Observation Space**: Technical indicators + portfolio state + sentiment scores
- **Reward Function**: Portfolio value change + risk penalties
- **Training**: 500K-1M timesteps typical

### Feature Engineering

**Technical Indicators**:
- Simple Moving Average (SMA 10, 20, 50, 200)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

**Sentiment Features**[4][7]:
- LLM-based news sentiment scoring
- Social media sentiment aggregation
- Financial report analysis
- Real-time news feed processing

### Environment Design

The custom `StockTradingEnv` implements:
- Realistic transaction costs and slippage[11]
- Multiple position management
- Portfolio rebalancing
- Market hours simulation
- Order execution modeling

---

## 📈 Results & Benchmarks

*Based on backtesting results from project assessment*

| Metric | Performance |
|--------|-------------|
| **Training Period** | 2020-2023 |
| **Test Period** | 2023-2024 |
| **Cumulative Return** | Varies by configuration |
| **Sharpe Ratio** | Competitive with index[2] |
| **Max Drawdown** | Risk-managed |

---

## 🔬 Research & Development

### Current Focus
- Fine-tuning LLM sentiment models for financial text[7][10]
- Implementing risk-averse reward functions[2]
- Multi-asset portfolio optimization
- Real-time trading pipeline deployment

### Planned Enhancements
See `NEXT_STEPS.md` for the complete development roadmap including:
- Advanced risk management
- Multi-exchange support
- Ensemble model strategies
- Live trading integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- **Not Financial Advice**: This system does not provide financial advice
- **Use at Your Own Risk**: Trading involves substantial risk of loss
- **No Guarantees**: Past performance does not guarantee future results
- **Test Thoroughly**: Always paper trade extensively before live deployment
- **Regulatory Compliance**: Ensure compliance with local trading regulations

---

## 📚 References & Acknowledgments

This project leverages research and tools from:
- **Stable-Baselines3**: RL algorithm implementations[3][9]
- **Gymnasium**: RL environment framework
- **FinRL**: Financial RL research framework[2][12]
- **LLM Sentiment Analysis**: Financial text analysis research[4][7][10]

---

## 📧 Contact

**Author**: Aditya Gautam

- GitHub: [@Aditya-gautam21](https://github.com/Aditya-gautam21)
- Project Link: [https://github.com/Aditya-gautam21/quantTrader](https://github.com/Aditya-gautam21/quantTrader)

---

## 🌟 Acknowledgments

- Thanks to the Stable-Baselines3 team for excellent RL implementations
- Financial sentiment research community for LLM integration insights[4][7][19]
- FinRL framework for inspiration on trading environment design[2][12]

---

**Star ⭐ this repo if you find it useful!**