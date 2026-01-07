import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, data_dir=f"./raw_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_historical_data(self, tickers, start_date=None, end_date=datetime.now(), interval="1d"):
        if end_date is None:
            end_date = datetime.now().date()
            
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Validate ticker symbols to prevent path traversal
        validated_tickers = []
        for ticker in tickers:
            # Only allow alphanumeric characters and common ticker symbols
            if ticker and ticker.replace('.', '').replace('-', '').replace('^', '').isalnum() and len(ticker) <= 10:
                validated_tickers.append(ticker.upper())
            else:
                print(f"Invalid ticker symbol: {ticker}")
        
        if not validated_tickers:
            print("No valid ticker symbols provided")
            return None
            
        downloaded_data = {}

        print(f"Downloading data for {validated_tickers}")

        data = None
        for ticker in validated_tickers:
            try:
                print(f"Downloading {ticker}")

                current_data_str = str(datetime.now().date())
                ticker_dir = self.data_dir / current_data_str / ticker

                ticker_dir.mkdir(parents=True, exist_ok=True)

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
                
                filename = f"historical_{ticker}.csv"
                filepath = ticker_dir / filename
                data.to_csv(filepath)

                downloaded_data[ticker] = data
                print(f"Data saved to {filepath}")   
                
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
                continue
        
        # Return all downloaded data after processing all tickers
        if downloaded_data:
            return downloaded_data
        else:
            print("No data was downloaded for any ticker")
            return None
    
    def load_data(self, filename):
        filepath = self.data_dir / filename
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
       
    def get_latest_price(self, ticker):
        data = yf.download(ticker, period="1d", auto_adjust=False)
        return data['Close'].iloc[-1]

if __name__ == "__main__":
    #tickers_input = input("What ticker to download?")
    #tickers = [tickers_input]
    collector = MarketDataCollector()
    historical = collector.download_historical_data(
        "AAPL",
        start_date=datetime.now() -  timedelta(days = 730),
        interval='4h'
    )