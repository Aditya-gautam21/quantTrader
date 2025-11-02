import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, data_dir="backend\data\raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_historical_data(self, tickers, start_date, end_date, interval="1d"):

        print(f"Downloading {tickers} from {start_date} to {end_date}...")

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=True
        )

        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product
            ([tickers, data.columns])

        filename = f"historical_{start_date}_to_{end_date},csv"
        filepath = self.data_dir / filename
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
        print(f" Shape: {data.shape}, Rows: {len(data)}, Columns: {len(data.columns)}")

        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    def load_data(self, filename):
        filepath = self.data_dir / filename
        return pd.read_csv(filepath, index_col=0, parse_Dates=True)

    def get_latest_price(self, ticker):
        data = yf.download(ticker, period="1d")
        return data['Close'].iloc[-1]

if __name__ == "__main__":
    collector = MarketDataCollector()
    historical = collector.download_historical_data(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2021-10-30',
        end_date='2025-10-30',
        interval='1d'
    )

    print("\n Data Preview:")
    print(historical.head())
    print("\nData Info:")
    print(historical.info())

    print(f"\nPrice Range for AAPL: ${historical[('AAPL', 'Close')].min():.2f} - ${historical[('AAPL', 'Close')].max():.2f}")