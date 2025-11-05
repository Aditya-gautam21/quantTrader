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