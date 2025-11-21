import pandas as pd
from pathlib import Path
from datetime import datetime

class DataCombiner:
    def __init__(self, base_dir='./raw_data'):
        self.base_dir = Path(base_dir)

    def combine_data(self, ticker):
        current_date_str = str(datetime.now().date())
        
        ticker_dir = self.base_dir / current_date_str / ticker
        
        hist_path = ticker_dir / f"historical_{ticker}.csv"
        norm_path = ticker_dir / f"normalized_{ticker}.csv"
        combined_path = ticker_dir / f"combined_{ticker}.csv"
        
        if not hist_path.exists():
            print(f"Error: Could not find historical data at {hist_path}")
            return
            
        if not norm_path.exists():
            print(f"Error: Could not find normalized data at {norm_path}")
            return

        ohlcv = pd.read_csv(hist_path)
        norm_indicators = pd.read_csv(norm_path)

        ohlcv['Date'] = pd.to_datetime(ohlcv['Date'])
        norm_indicators['Date'] = pd.to_datetime(norm_indicators['Date'])
        
        merged_df = pd.merge(ohlcv, norm_indicators, on='Date', how='inner')

        merged_df.to_csv(combined_path, index=False)
        
        print(f"Merged shape: {merged_df.shape}")
        print(f"Data combined and saved to: {combined_path}")

if __name__ == "__main__":
    ticker = input("Enter ticker to combine: ")
    combiner = DataCombiner()
    combiner.combine_data(ticker)