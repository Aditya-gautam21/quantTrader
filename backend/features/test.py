import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from datetime import datetime

class TechnicalIndicators:
    def __init__(self, data_dir="./raw_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_indicators(data, debug=False):
        print(f"\n Calculating technical indicators")

        df = data.copy()
        df.sort_index(inplace = True)

        open_ = df['Open']
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        indicators = pd.DataFrame(index=df.index)
        
        # 2. Momentum Indicators
        indicators['RSI_14'] = ta.rsi(close, length=14)
        MACD = ta.macd(close)
        indicators['MACD_HIST'] = MACD['MACDh_12_26_9']
        
        # 3. Volatility Indicators
        bb = ta.bbands(close, length=20, std=2)

        bb_lower = bb.iloc[:, 0]  
        bb_middle = bb.iloc[:, 1] 
        bb_upper = bb.iloc[:, 2]   

        indicators['BB_POSITION'] = (close - bb_lower) / (bb_upper - bb_lower)

        # 6. Returns
        indicators['RET_1'] = np.log(close / close.shift(1))
        indicators['RET_5'] = np.log(close / close.shift(5))
        indicators['RET_15'] = np.log(close / close.shift(15))

        # 7. Trend - THIS IS THE CRITICAL SECTION
        EMA21 = ta.ema(close, length=21)
        
        if debug:
            print("\n" + "="*80)
            print("DEBUG: EMA21 CALCULATIONS")
            print("="*80)
            print(f"\nClose prices (first 25):\n{close.head(25)}")
            print(f"\nEMA21 values (first 25):\n{EMA21.head(25)}")
            print(f"\nEMA21.diff() (first 25):\n{EMA21.diff().head(25)}")
            print(f"\n(close - EMA21) (first 25):\n{(close - EMA21).head(25)}")
        
        indicators['EMA21_SLOPE'] = EMA21.diff() 
        indicators['PRICE_EMA21_DIST'] = close - EMA21

        # 8, Volatility
        atr = ta.atr(high, low, close)
        indicators['ATR'] = atr
        indicators['RET_STD'] = indicators['RET_1'].rolling(60).std()

        # 9. Volume
        indicators['VOL'] = volume
        vwap = ta.vwap(high, low, close, volume)

        if vwap is not None:
            indicators['VWAP_DIST'] = close - vwap
        else:
            indicators['VWAP_DIST'] = np.nan

        # 10. Candle microstructure
        indicators['BODY_ATR'] = (close - open_) / atr
        indicators['UPPER_WICK_ATR'] = (high - np.maximum(close, open_)) / atr
        indicators['LOWER_WICK_ATR'] = (np.minimum(close, open_) - low) / atr

        if debug:
            print("\n" + "="*80)
            print("DEBUG: BEFORE DROPNA")
            print("="*80)
            print(f"Total rows: {len(indicators)}")
            print(f"EMA21_SLOPE NaN count: {indicators['EMA21_SLOPE'].isna().sum()}")
            print(f"PRICE_EMA21_DIST NaN count: {indicators['PRICE_EMA21_DIST'].isna().sum()}")
            print(f"\nFirst 25 rows of both columns:")
            print(indicators[['EMA21_SLOPE', 'PRICE_EMA21_DIST']].head(25))

        result = indicators.dropna()
        
        if debug:
            print("\n" + "="*80)
            print("DEBUG: AFTER DROPNA")
            print("="*80)
            print(f"Total rows after dropna: {len(result)}")
            print(f"\nFirst 10 rows:")
            print(result[['EMA21_SLOPE', 'PRICE_EMA21_DIST']].head(10))
            
            print("\n" + "="*80)
            print("DEBUG: PROPORTIONALITY CHECK")
            print("="*80)
            ratio = result['PRICE_EMA21_DIST'] / result['EMA21_SLOPE']
            print(f"Ratio (PRICE_EMA21_DIST / EMA21_SLOPE) - first 10:")
            print(ratio.head(10))
            print(f"\nRatio statistics:")
            print(ratio.describe())
            print(f"\nNumber of unique ratios: {ratio.nunique()}")
            print(f"Is ratio constant (within tolerance)? {ratio.std() < 1e-6}")
            
            if ratio.std() < 1e-6:
                print(f"\n⚠️  WARNING: RATIO IS CONSTANT = {ratio.iloc[0]:.6f}")
                print("This means PRICE_EMA21_DIST is a perfect scalar multiple of EMA21_SLOPE!")
            else:
                print(f"\n✓ GOOD: Ratio varies (std = {ratio.std():.6f})")
            
            print("\n" + "="*80)
            print("DEBUG: CORRELATION CHECK")
            print("="*80)
            correlation = result['EMA21_SLOPE'].corr(result['PRICE_EMA21_DIST'])
            print(f"Correlation between EMA21_SLOPE and PRICE_EMA21_DIST: {correlation:.6f}")
            
            if correlation > 0.99:
                print("⚠️  WARNING: Nearly perfect correlation!")
            else:
                print(f"✓ Correlation is reasonable")

        print(f"\nAdded {len(result.columns)} technical indicators")
        print(f"Indicators columns: {result.columns.tolist()}\n")

        return result
        
    def save_indicators(self, df, symbol):
        current_data_str = str(datetime.now().date())

        symbol_dir = self.data_dir / current_data_str
        symbol_dir.mkdir(parents=True, exist_ok=True)
        prefix = "indicators"
                
        filename = f"{prefix}_{symbol}.csv"
        filepath = symbol_dir / filename
            
        df.to_csv(filepath)
        print(f"Saved data to: {filepath}")
        
        # Verification: reload and check
        print("\n" + "="*80)
        print("VERIFICATION: Reading saved file")
        print("="*80)
        loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"\nFirst 5 rows of saved data:")
        print(loaded[['EMA21_SLOPE', 'PRICE_EMA21_DIST']].head())
        
        ratio_saved = loaded['PRICE_EMA21_DIST'] / loaded['EMA21_SLOPE']
        print(f"\nRatio in saved file (first 5): {ratio_saved.head().tolist()}")
        print(f"Ratio std in saved file: {ratio_saved.std():.6f}")

if __name__ == "__main__":
    # Load data
    data_path = "raw_data/2026-01-11/BTCUSDT/historical_BTCUSDT.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.index.name = "Timestamp"    

    print("="*80)
    print("RUNNING TECHNICAL INDICATORS WITH DEBUG MODE")
    print("="*80)
    
    # Calculate indicators with debug=True
    indicators = TechnicalIndicators.calculate_indicators(data, debug=True)
    
    # Save and verify
    TechnicalIndicators().save_indicators(indicators, symbol='BTCUSDT_DEBUG')
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nIf the ratio is constant (~10), the bug is in the calculation.")
    print("If the ratio varies normally but the saved file shows constant ratio,")
    print("the bug is in the save/load process.")