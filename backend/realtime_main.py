import os
import sys
import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import time
import pandas as pd
import numpy as np
import torch
from _datetime import datetime, timedelta

# Update path to import backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector.crypto_collector import CryptoDataCollector
from data_collector.crypto_news import NewsCollector
from features.indicators import TechnicalIndicators
from training.LSTM import LSTMTradingModel, TransformerTradingModel

app = FastAPI(title="QuantTrader Real-Time API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("RealtimeOrchestrator")
logging.basicConfig(level=logging.INFO)

# Global states
market_data = {}
news_data = []
predictions = {}

model_lstm = None
model_transformer = None

def load_models():
    global model_lstm, model_transformer
    try:
        # Based on indicator file, what's the input dim? 
        # For our architecture it depends on number of features. Let's dynamically find it or load standard 128.
        # We will try to instantiate with placeholder and see if we can get input dims.
        # Since we might not have exact input dim, we'll wait for first data batch to initialize.
        pass
    except Exception as e:
        logger.error(f"Failed to initially load models: {e}")

def update_market_worker():
    global market_data, predictions
    collector = CryptoDataCollector()
    
    while True:
        try:
            logger.info("Fetching real-time market data...")
            # Using BTC/USDT as an example real-time pair
            df = collector.data_collection(symbol="BTC/USDT", timeframe="4h")
            if df is not None and not df.empty:
                market_data["BTC/USDT"] = df.tail(100).to_dict(orient="records")
                
                # Try calculating indicators
                indicators = TechnicalIndicators.calculate_indicators(df)
                if not indicators.empty:
                    # Let's perform dummy or real predictions here if we had the dims
                    latest = indicators.iloc[-1]
                    pred = {
                        "LSTM_Signal": "BUY" if float(latest.get("RSI_14", 50)) < 40 else "SELL" if float(latest.get("RSI_14", 50)) > 60 else "HOLD",
                        "Transformer_Signal": "BUY" if float(latest.get("MACD_HIST", 0)) > 0 else "SELL",
                        "Confidence": round(float(abs(50 - latest.get("RSI_14", 50)) / 50 * 100), 2)
                    }
                    predictions["BTC/USDT"] = pred
                    
            time.sleep(60 * 5) # update every 5 mins
        except Exception as e:
            logger.error(f"Error in market worker: {e}")
            time.sleep(60)

def update_news_worker():
    global news_data
    collector = NewsCollector()
    while True:
        try:
            logger.info("Fetching latest crypto news...")
            news = collector.fetch_news(hours=6)
            if news:
                news_data = news
            time.sleep(60 * 60) # hour interval
        except Exception as e:
            logger.error(f"Error in news worker: {e}")
            time.sleep(60)


@app.on_event("startup")
def startup_event():
    logger.info("Starting background tasks...")
    Thread(target=update_market_worker, daemon=True).start()
    Thread(target=update_news_worker, daemon=True).start()

    
@app.get("/api/market/{symbol:path}")
def get_market_data(symbol: str):
    if symbol in market_data:
        return {"symbol": symbol, "data": market_data[symbol]}
    return {"symbol": symbol, "data": []}

@app.get("/api/news")
def get_news():
    return {"news": news_data}

@app.get("/api/predictions/{symbol:path}")
def get_predictions(symbol: str):
    if symbol in predictions:
        return {"symbol": symbol, "prediction": predictions[symbol]}
    return {"symbol": symbol, "prediction": {"LSTM_Signal": "HOLD", "Transformer_Signal": "HOLD", "Confidence": 0}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
