import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime

class TechnicalIndicators:
    def __init__(self, data_dir="./raw_data", norm_data_dir="./raw_data/normalized_indicators"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.norm_data_dir = Path(norm_data_dir)
        self.norm_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_indicators(df):
        """
        Adding technial indicators to OHLCV data
        """

        print(f"\n Calculating technical indicators")

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        indicators = pd.DataFrame(index=df.index)

        #1 Moving Averages
        indicators['SMA_20'] = ta.sma(close, length=20)
        indicators['SMA_50'] = ta.sma(close, length=50)
        indicators['EMA_12'] = ta.ema(close, length=12)
        indicators['EMA_26'] = ta.ema(close, length=26)
        
        # 2. Momentum Indicators
        indicators['RSI_14'] = ta.rsi(close, length=14)
        indicators['MACD'] = ta.macd(close)['MACD_12_26_9']
        indicators['MACD_SIGNAL'] = ta.macd(close)['MACDs_12_26_9']
        
        # 3. Volatility Indicators
        bb = ta.bbands(close, length=20)
        if bb is not None:
            indicators['BB_UPPER'] = bb['BBU_20_2.0_2.0']
            indicators['BB_LOWER'] = bb['BBL_20_2.0_2.0']
            indicators['BB_MIDDLE'] = bb['BBM_20_2.0_2.0']
        else:
            indicators['BB_UPPER'] = 0
            indicators['BB_LOWER'] = 0
            indicators['BB_MIDDLE'] = 0
        
        # 4. Trend Indicators
        adx = ta.adx(high, low, close)
        indicators['ADX_14'] = adx['ADX_14']
        
        # 5. Volume Indicators
        indicators['OBV'] = ta.obv(close, volume)

        indicators = indicators.bfill().ffill()

        print(f"Addedd {len(indicators.columns)} technical indicators")
        print(f"Indicators columns: {indicators.columns.tolist()}\n")

        return indicators
    
    @staticmethod
    def normalize_indicators(df):
        norm = df.copy()

        if 'RSI_14' in norm.columns:
            norm['RSI_14'] = norm['RSI_14'] / 100.0
            
        if 'ADX_14' in norm.columns:
            norm['ADX_14'] = norm['ADX_14'] / 100.0

        unbounded_cols = ['MACD', 'MACD_SIGNAL', 'OBV', 'CCI']
        
        for col in unbounded_cols:
            if col in norm.columns:
                rolling_mean = norm[col].rolling(window=60, min_periods=1).mean()
                rolling_std = norm[col].rolling(window=60, min_periods=1).std()
                norm[col] = ((norm[col] - rolling_mean) / (rolling_std + 1e-8)).clip(-3, 3)
                
            
        price_col = 'Close' 
        if price_col in norm.columns:
            if 'SMA_20' in norm.columns:
                norm['SMA_20'] = norm['Close'] / norm['SMA_20'] 
            
            if 'BB_UPPER' in norm.columns:
                norm['BB_POSITION'] = (norm['Close'] - norm['BB_LOWER']) / (norm['BB_UPPER'] - norm['BB_LOWER'])
                
                norm.drop(columns=['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE'], inplace=True)

        return norm
    
    def save_indicators(self, df, ticker, is_normalized=False):
        current_data_str = str(datetime.now().date())

       

        if is_normalized:
            # Save to raw/normalized_indicators
            ticker_dir = self.data_dir / current_data_str / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            folder = self.norm_data_dir
            prefix = "normalized"
        else:
             # Save to raw/indicators
            ticker_dir = self.data_dir / current_data_str / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            folder = self.data_dir
            prefix = "indicators"
                
        filename = f"{prefix}_{ticker}.csv"
        filepath = ticker_dir / filename
            
        df.to_csv(filepath)
        print(f"Saved {'normalized ' if is_normalized else ''}data to: {filepath}")

if __name__ == "__main__":
    from data_collector.market_collector import MarketDataCollector

    # Load data
    data_path = "raw_data/2026-01-07/BTC_USDT/historical_BTC_USDT.csv"
    data = pd.read_csv(data_path)
    '''collector = MarketDataCollector()
    data_dict = collector.download_historical_data(
        tickers=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-10-30'
    )

    if data_dict and 'AAPL' in data_dict:
        data = data_dict['AAPL']'''
    
    # Calculate indicators
    indicators = TechnicalIndicators.calculate_indicators(data)
    norm_indicators = TechnicalIndicators.normalize_indicators(indicators)

    TechnicalIndicators().save_indicators(indicators, ticker='BTCUSDT', is_normalized=False)
    TechnicalIndicators().save_indicators(norm_indicators, ticker='BTCUSDT', is_normalized=True)