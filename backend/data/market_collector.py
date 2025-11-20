import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, data_dir="./raw/historical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_historical_data(self, tickers, start_date, end_date, interval="1d"):
        if end_date is None:
            end_date = datetime.now().date()
            
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        if isinstance(tickers, str):
            tickers = [tickers]
            
        downloaded_data = {}

        print(f"Downloading data for {tickers}")

        data = None
        for ticker in tickers:
            try:
                print(f"Downloading {ticker}")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    multi_level_index=False
                )

                if data is None or data.empty:
                    print(f"No data found for {ticker}")
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)

                data.index.name = 'Date'
                
                filename = f"historical_{ticker}_{start_date}_to_{end_date}.csv"
                filepath = self.data_dir / filename
                data.to_csv(filepath)

                downloaded_data[ticker] = data
            
                print(f"Data saved to {filepath}")                
                
            except Exception as e:
                print(f"Error downloading data: {e}")
                return None
            
        return downloaded_data
    
    def load_data(self, filename):
        filepath = self.data_dir / filename
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
       
    def get_latest_price(self, ticker):
        data = yf.download(ticker, period="1d", auto_adjust=False)
        return data['Close'].iloc[-1]

if __name__ == "__main__":
    tickers_input = input("What ticker to download?")
    tickers = [tickers_input]
    collector = MarketDataCollector()
    historical = collector.download_historical_data(
        tickers,
        start_date='2021-10-30',
        end_date='2025-10-30',
        interval='1d'
    )

    if historical is not None:
        print("\nData Preview:")
        print(historical.head())
        print("\nData Info:")
        print(historical.info())