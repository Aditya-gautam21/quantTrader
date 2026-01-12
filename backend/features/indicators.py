import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime

class TechnicalIndicators:
    def __init__(self, data_dir="./raw_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_indicators(data):
        print(f"\n Calculating technical indicators")

        df = data.copy()
        df.sort_index(inplace = True)

        open_ = df['Open']
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        indicators = pd.DataFrame(index=df.index)

        '''#1 Moving Averages
        indicators['SMA_20'] = ta.sma(close, length=20)
        indicators['SMA_50'] = ta.sma(close, length=50)
        indicators['EMA_12'] = ta.ema(close, length=12)
        indicators['EMA_26'] = ta.ema(close, length=26)'''
        
        # 2. Momentum Indicators
        indicators['RSI_14'] = ta.rsi(close, length=14)
        MACD = ta.macd(close)
        indicators['MACD_HIST'] = MACD['MACDh_12_26_9']
        
        # 3. Volatility Indicators
        bb = ta.bbands(close, length=20, std=2)

        bb_lower = bb.iloc[:, 0]  
        bb_middle = bb.iloc[:, 1] 
        bb_upper = bb.iloc[:, 2]   

        indicators['BB_POSITION'] = (close - bb_lower) / (bb_upper - bb_lower)

        '''# 4. Trend Indicators
        adx = ta.adx(high, low, close)
        indicators['ADX_14'] = adx['ADX_14']
        
        # 5. Volume Indicators
        indicators['OBV'] = ta.obv(close, volume)'''

        # 6. Returns
        indicators['RET_1'] = np.log(close / close.shift(1))
        indicators['RET_5'] = np.log(close / close.shift(5))
        indicators['RET_15'] = np.log(close / close.shift(15))

        # 7. Trend
        EMA21 = ta.ema(close, length=21)
        indicators['EMA21_SLOPE'] = EMA21.diff() 
        indicators['PRICE_EMA21_DIST'] = close - EMA21

        # 8, Volatility
        atr = ta.atr(high, low, close)
        indicators['ATR'] = atr
        indicators['RET_STD'] = indicators['RET_1'].rolling(60).std()

        # 9. Volume
        indicators['VOL'] = volume
        vwap = ta.vwap(high, low, close, volume)

        if vwap is not None:
            indicators['VWAP_DIST'] = close - vwap
        else:
            indicators['VWAP_DIST'] = np.nan


        # 10. Candle microstructre
        indicators['BODY_ATR'] = (close - open_) / atr
        indicators['UPPER_WICK_ATR'] = (high - np.maximum(close, open_)) / atr
        indicators['LOWER_WICK_ATR'] = (np.minimum(close, open_) - low) / atr


        print(f"Addedd {len(indicators.columns)} technical indicators")
        print(f"Indicators columns: {indicators.columns.tolist()}\n")

        return indicators.dropna()
        
    def save_indicators(self, df, symbol):
        current_data_str = str(datetime.now().date())

        # Save to raw/indicators
        symbol_dir = self.data_dir / current_data_str
        symbol_dir.mkdir(parents=True, exist_ok=True)
        folder = self.data_dir
        prefix = "indicators"
                
        filename = f"{prefix}_{symbol}.csv"
        filepath = symbol_dir / filename
            
        df.to_csv(filepath)
        print(f"Saved data to: {filepath}")

if __name__ == "__main__":
    from data_collector.market_collector import MarketDataCollector

    # Load data
    data_path = "raw_data/2026-01-11/BTCUSDT/historical_BTCUSDT.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"    

    # Calculate indicators
    indicators = TechnicalIndicators.calculate_indicators(data)
    TechnicalIndicators().save_indicators(indicators, symbol='BTCUSDT')