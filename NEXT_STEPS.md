# Next Steps for QuantTrader

## 🐛 **Issues Found**

### 1. **Bug in `main.py` (Line 54)**
**Problem:** `calculate_indicators` is called with `ticker` parameter, but the function signature doesn't accept it.
```python
# Current (incorrect):
indicators = TechnicalIndicators.calculate_indicators(market_data, ticker='AAPL')

# Should be:
# market_data is a dict, need to extract DataFrame first
if 'AAPL' in market_data:
    indicators = TechnicalIndicators.calculate_indicators(market_data['AAPL'])
```

### 2. **Bug in `main.py` (Line 27-31)**
**Problem:** `download_historical_data` returns early after first ticker, so `MSFT` is never downloaded.
- The function returns inside the loop after processing first ticker
- Should collect all tickers before returning

### 3. **Issue in `aggregator.py` (Line 14)**
**Problem:** `LlamaSentimentAnalyzer` constructor doesn't accept `model` parameter.
```python
# Current (incorrect):
self.sentiment_analyzer = LlamaSentimentAnalyzer(model="llama3.2:1b")

# Should be:
self.sentiment_analyzer = LlamaSentimentAnalyzer()
```

### 4. **Issue in `aggregator.py` (Line 48)**
**Problem:** Accessing `ohlcv_data` with tuple key `(ticker, 'Close')` may not work if data structure is different.
- Need to handle both dict and DataFrame formats

---

## 🎯 **Immediate Action Items**

### **Phase 1: Fix Critical Bugs** (1-2 hours)
1. ✅ Fix `main.py` to properly handle market_data dictionary
2. ✅ Fix `market_collector.py` to return all tickers
3. ✅ Fix `aggregator.py` sentiment analyzer initialization
4. ✅ Test the full pipeline end-to-end

### **Phase 2: Build Backtesting Framework** (1-2 days)
**Priority: CRITICAL** - This is the most important missing piece.

**Files to create:**
```
backend/backtest/
├── __init__.py
├── backtest_engine.py      # Main backtesting engine
├── metrics.py              # Performance metrics calculation
└── visualizer.py           # Charts and visualizations
```

**Key features:**
- Replay historical data through trained model
- Calculate Sharpe ratio, Sortino ratio, max drawdown
- Compare with buy-and-hold baseline
- Generate equity curve and trade visualization
- Export results to CSV/JSON

### **Phase 3: Create API Server** (1 day)
**Priority: HIGH** - Enables frontend and automation

**Files to create:**
```
backend/api/
├── __init__.py
├── main.py                 # FastAPI app
├── routes/
│   ├── __init__.py
│   ├── data.py            # Data collection endpoints
│   ├── training.py        # Model training endpoints
│   ├── trading.py         # Paper trading endpoints
│   └── analytics.py       # Performance metrics endpoints
└── models.py              # Pydantic schemas
```

**Endpoints needed:**
- `POST /api/data/collect` - Trigger data collection
- `POST /api/train` - Start model training
- `POST /api/paper-trade` - Run paper trading
- `GET /api/performance` - Get performance metrics
- `GET /api/trades` - Get trade history

### **Phase 4: Build Frontend Dashboard** (2-3 days)
**Priority: HIGH** - Visualization and monitoring

**Tech stack:**
- React + TypeScript
- Chart.js or Recharts for visualizations
- Tailwind CSS for styling

**Components needed:**
- Portfolio value chart (real-time)
- Trade history table
- Performance metrics cards
- Strategy configuration form
- Model training status

### **Phase 5: Add Database** (1 day)
**Priority: MEDIUM** - Better data persistence

**Options:**
- SQLite for development (simple, no setup)
- PostgreSQL for production (better performance)

**Tables:**
- `trades` - All executed trades
- `performance_snapshots` - Daily performance metrics
- `model_versions` - Trained model metadata
- `configurations` - Strategy configurations

---

## 📋 **Recommended Development Order**

### **Week 1: Foundation**
1. **Day 1:** Fix bugs in existing code
2. **Day 2-3:** Build backtesting framework
3. **Day 4:** Test backtesting with existing models
4. **Day 5:** Create API server (basic endpoints)

### **Week 2: Frontend & Integration**
1. **Day 1-2:** Build React dashboard (basic version)
2. **Day 3:** Connect frontend to API
3. **Day 4:** Add database integration
4. **Day 5:** Testing and bug fixes

### **Week 3: Enhancement**
1. **Day 1-2:** Add risk management module
2. **Day 3:** Real-time data streaming
3. **Day 4:** Enhanced monitoring
4. **Day 5:** Documentation and deployment prep

---

## 🚀 **Quick Wins (Can Do Today)**

1. **Fix the bugs** mentioned above (30 minutes)
2. **Create a simple backtesting script** (2-3 hours)
   - Load trained model
   - Run on historical data
   - Calculate basic metrics (total return, Sharpe ratio)
   - Plot equity curve

3. **Add a simple FastAPI server** (1-2 hours)
   - Single endpoint to run paper trading
   - Return JSON with results

4. **Create a basic HTML dashboard** (2-3 hours)
   - Simple static page with Chart.js
   - Display portfolio value over time
   - Show recent trades

---

## 📊 **Success Criteria**

### **Minimum Viable Product (MVP)**
- ✅ Data collection works
- ✅ Model training works
- ✅ Paper trading works
- ⚠️ **Backtesting works** (NEEDS IMPLEMENTATION)
- ⚠️ **Basic dashboard** (NEEDS IMPLEMENTATION)
- ⚠️ **API endpoints** (NEEDS IMPLEMENTATION)

### **Production Ready**
- All MVP features
- Database integration
- Risk management
- Real-time monitoring
- Error handling and logging
- Documentation
- Testing suite

---

## 💡 **Tips**

1. **Start small:** Get backtesting working first - it's the most critical missing piece
2. **Iterate:** Build a simple version, test it, then enhance
3. **Test thoroughly:** Backtest on multiple time periods and stocks
4. **Monitor performance:** Track metrics over time to improve strategies
5. **Document as you go:** Write docstrings and comments

---

**Ready to start?** I recommend beginning with:
1. Fixing the bugs
2. Building a simple backtesting script
3. Creating a basic API endpoint

Would you like me to help implement any of these?

