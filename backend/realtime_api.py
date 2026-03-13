import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantTrader")

# Global state
market_data = {}
news_data = []
predictions = {}
models_loaded = False
model_lstm = None
model_transformer = None

FEATURE_COLS = [
    "RSI_14", "MACD_HIST", "BB_POSITION", "RET_1", "RET_5", "RET_15",
    "PRICE_EMA21_DIST", "ATR", "RET_STD", "VOL", "VWAP_DIST",
    "BODY_ATR", "UPPER_WICK_ATR", "LOWER_WICK_ATR"
]

def load_models():
    global model_lstm, model_transformer, models_loaded
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_dim = len(FEATURE_COLS)
        
        model_lstm = LSTMTradingModel(input_dim=input_dim)
        model_transformer = TransformerTradingModel(input_dim=input_dim)
        
        lstm_path = os.path.join(os.path.dirname(__file__), 'models', 'best_lstm.pth')
        transformer_path = os.path.join(os.path.dirname(__file__), 'models', 'best_transformer.pth')
        
        if os.path.exists(lstm_path):
            model_lstm.load_state_dict(torch.load(lstm_path, map_location=device))
            model_lstm.eval()
            logger.info("✓ LSTM model loaded")
        
        if os.path.exists(transformer_path):
            model_transformer.load_state_dict(torch.load(transformer_path, map_location=device))
            model_transformer.eval()
            logger.info("✓ Transformer model loaded")
        
        models_loaded = True
    except Exception as e:
        logger.error(f"Model loading error: {e}")

def predict_signal(indicators_df):
    """Generate trading signals from models"""
    if not models_loaded or indicators_df.empty:
        return {"LSTM_Signal": "HOLD", "Transformer_Signal": "HOLD", "Confidence": 0}
    
    try:
        # Get latest features
        latest = indicators_df[FEATURE_COLS].iloc[-60:].values
        if len(latest) < 60:
            latest = np.pad(latest, ((60-len(latest), 0), (0, 0)), mode='edge')
        
        x = torch.FloatTensor(latest).unsqueeze(0)
        
        with torch.no_grad():
            lstm_out = model_lstm(x).squeeze().numpy()
            trans_out = model_transformer(x).squeeze().numpy()
        
        lstm_signal = ["SELL", "HOLD", "BUY"][np.argmax(lstm_out)]
        trans_signal = ["SELL", "HOLD", "BUY"][np.argmax(trans_out)]
        
        confidence = (np.max(lstm_out) + np.max(trans_out)) / 2 * 100
        
        return {
            "LSTM_Signal": lstm_signal,
            "Transformer_Signal": trans_signal,
            "Confidence": round(float(confidence), 2)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"LSTM_Signal": "HOLD", "Transformer_Signal": "HOLD", "Confidence": 0}

def update_market_worker():
    global market_data, predictions
    collector = CryptoDataCollector()
    
    while True:
        try:
            logger.info("Fetching market data...")
            df = collector.data_collection(symbol="BTC/USDT", timeframe="4h")
            
            if df is not None and not df.empty:
                market_data["BTC/USDT"] = df.tail(100).reset_index().to_dict(orient="records")
                
                indicators = TechnicalIndicators.calculate_indicators(df)
                if not indicators.empty:
                    predictions["BTC/USDT"] = predict_signal(indicators)
                    
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Market worker error: {e}")
            time.sleep(60)

def update_news_worker():
    global news_data
    collector = NewsCollector()
    
    while True:
        try:
            logger.info("Fetching news...")
            news = collector.fetch_news(hours=6)
            if news:
                news_data = news[:10]  # Keep latest 10
            time.sleep(3600)  # 1 hour
        except Exception as e:
            logger.error(f"News worker error: {e}")
            time.sleep(60)

@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting QuantTrader API...")
    load_models()
    Thread(target=update_market_worker, daemon=True).start()
    Thread(target=update_news_worker, daemon=True).start()

@app.get("/")
def root():
    return {"status": "online", "service": "QuantTrader API"}

@app.get("/api/market/{symbol:path}")
def get_market_data(symbol: str):
    return {"symbol": symbol, "data": market_data.get(symbol, [])}

@app.get("/api/news")
def get_news():
    return {"news": news_data}

@app.get("/api/predictions/{symbol:path}")
def get_predictions(symbol: str):
    default = {"LSTM_Signal": "HOLD", "Transformer_Signal": "HOLD", "Confidence": 0}
    return {"symbol": symbol, "prediction": predictions.get(symbol, default)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
