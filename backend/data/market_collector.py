import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, data_dir="backend\data\raw"):