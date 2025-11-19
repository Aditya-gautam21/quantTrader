import pandas as pd
import numpy as np
from indicators import TechnicalIndicators

hist_data = pd.read_csv('raw/historical_2023-01-01_to_2024-10-30.csv')
data_df = pd.DataFrame()
indicator = TechnicalIndicators.calculate_indicators(hist_data, ticker='AAPL')
data_df['PRICE'] = hist_data['Adj Close']