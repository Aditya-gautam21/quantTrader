import pandas as pd
from pathlib import Path
from datetime import datetime
from data_collector.crypto_collector import CryptoDataCollector
from features.indicators import TechnicalIndicators
from features.norm_indicators import NormalizedIndicators

class AutoNormaliser:
    def __init__(self, base_dir='./raw_data'):
        self.base_dir = Path(base_dir)

    def auto_normalisation(self):
        collector = CryptoDataCollector()
        indicator = TechnicalIndicators()
        normaliser = NormalizedIndicators()

        data = collector.data_collection(symbol="BTC/USDT")
        raw_indicators = indicator.calculate_indicators(data)
        normalised_indicators = normaliser.normalize_indicators(raw_indicators)

        normaliser.save_indicators(normalised_indicators, symbol="BTCUSDT")


if __name__ == "__main__":
  combiner = AutoNormaliser()

  combiner.auto_normalisation()