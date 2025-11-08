import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalIndicators:
    
    @staticmethod
    def calculate_indicators(df, ticker=None):
        """
        Adding technial indicators to OHLCV data
        """

        print(f"\n Calculating technical indicators")

        if ticker:
            close = df[(ticker, 'Close')]
            high = df[(ticker, 'High')]
            low = df[(ticker, 'Low')]
            volume = df[(ticker, 'Volume')]
        else:
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
        indicators['MACD_SIGNAL'] = ta.macd(close)['MACDh_12_26_9']
        
        # 3. Volatility Indicators
        bb = ta.bbands(close, length=20)
        indicators['BB_UPPER'] = bb['BBU_20_2.0']
        indicators['BB_LOWER'] = bb['BBL_20_2.0']
        indicators['BB_MIDDLE'] = bb['BBM_20_2.0']
        
        # 4. Trend Indicators
        indicators['ADX_14'] = ta.adx(high, low, close)['ADX_14']
        
        # 5. Volume Indicators
        indicators['OBV'] = ta.obv(close, volume)

        indicators = indicators.fillna(method='bfill').fillna(method='ffill')

        print(f"Addedd {len(indicators.columns)} technical indicators")
        print(f"Indicators columns: {indicators.columns.tolist()}\n")

        return indicators
    
    @staticmethod
    def normalize_indicators(indicators_df):
        """Normalize indicators to [0, 1] range for RL agent"""
        normalized = indicators_df.copy()
        
        for col in normalized.columns:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.5
        
        return normalized

# LEARNING EXERCISE
if __name__ == "__main__":
    from data.market_collector import MarketDataCollector
    
    # Load data
    collector = MarketDataCollector()
    data = collector.download_historical_data(
        tickers=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-10-30'
    )
    
    # Calculate indicators
    indicators = TechnicalIndicators.calculate_indicators(data, ticker='AAPL')
    
    print("\n📊 Indicator Preview:")
    print(indicators.tail())