import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class CryptoDataCollector:
    def __init__(self, data_dir="./raw_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def data_collection(self, symbol, timeframe="4h", since=None):
        exchange = ccxt.binance({
            'enableRateLimit': True
        })

        if since is None:
            since = datetime.now() - timedelta(days=365 * 5)

        # convert to milliseconds timestamp
        since_ms = exchange.parse8601(
            since.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        all_data = []

        while True:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=1000
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            since_ms = ohlcv[-1][0] + 1

        if not all_data:
            print("No data fetched.")
            return None

        df = pd.DataFrame(
            all_data,
            columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        )

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=["Timestamp"], inplace=True)

        # clean data
        df = df[~df.index.duplicated()]
        df = df.sort_index()

        current_date = datetime.now().date().isoformat()
        symbol_dir = self.data_dir / current_date / symbol.replace("/", "_")
        symbol_dir.mkdir(parents=True, exist_ok=True)

        filepath = symbol_dir / f"historical_{symbol.replace('/', '_')}.csv"
        df.to_csv(filepath)

        print(f"✅ Data saved to {filepath}")
        print(f"📊 Rows: {len(df)} | From {df.index.min()} to {df.index.max()}")

        return df


if __name__ == "__main__":
    collector = CryptoDataCollector()

    data = collector.data_collection(
        symbol="BTC/USDT",
        timeframe="4h",
        since=datetime.now() - timedelta(days=365 * 5)
    )
