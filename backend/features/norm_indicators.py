import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime

class NormalizedIndicators:
    def __init__(self, data_dir="./raw_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def normalize_indicators(self, df, window=60):


        norm = df.copy()

        if 'RSI_14' in norm:
            norm['RSI_14'] = (norm['RSI_14'] - 50) / 50

        if 'BB_POSITION' in norm:
            norm['BB_POSITION'] = (norm['BB_POSITION'] - 0.5) * 2

        unbounded = [
            'RET_1','RET_5','RET_15',
            'EMA21_SLOPE','PRICE_EMA21_DIST',
            'MACD_HIST','ATR','RET_STD',
            'VOL','VWAP_DIST'
        ]

        for col in unbounded:
            mean = norm[col].rolling(window).mean()
            std = norm[col].rolling(window).std()
            norm[col] = ((norm[col] - mean) / (std + 1e-8)).clip(-5, 5)

        return norm.dropna()
    
    def save_indicators(self, norm, symbol):
        current_data_str = str(datetime.now().date())

        symbol_dir = self.data_dir / current_data_str
        symbol_dir.mkdir(parents=True, exist_ok=True)
        prefix = "norm_indicators"
                
        filename = f"{prefix}_{symbol}.csv"
        filepath = symbol_dir / filename
            
        norm.to_csv(filepath)
        print(f"Saved data to: {filepath}")

if __name__ == '__main__':
    data_path = "raw_data/2026-01-11/indicators_BTCUSDT.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"    

    Normaliser = NormalizedIndicators()
    
    # Calculate indicators
    indicators = Normaliser.normalize_indicators(data, window=60)
    Normaliser.save_indicators(indicators, symbol='BTCUSDT')