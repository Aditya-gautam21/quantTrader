"""
Combine all features into state vector for RL agent
"""

import pandas as pd
import numpy as np
from features.indicators import TechnicalIndicators
from features.sentiment import LlamaSentimentAnalyzer

class FeatureAggregator:
    """Combine technical indicators + sentiment into unified feature vector"""
    
    def __init__(self):
        self.sentiment_analyzer = LlamaSentimentAnalyzer()
        self.indicators = TechnicalIndicators()
    
    def create_state_features(self, ohlcv_data, news_data, ticker='AAPL'):
        """
        Create complete feature vector for RL agent
        
        Args:
            ohlcv_data: DataFrame with price data
            news_data: List of recent news items
            ticker: Ticker symbol
        
        Returns:
            DataFrame with all features combined
        """
        print("\n🔗 Aggregating all features...")
        
        # Handle ohlcv_data - could be dict or DataFrame
        if isinstance(ohlcv_data, dict):
            if not ohlcv_data:
                raise ValueError("Empty OHLCV data dictionary provided")
            # If dict, extract DataFrame for the ticker
            if ticker in ohlcv_data:
                data_df = ohlcv_data[ticker]
            else:
                # Use first available ticker
                data_df = list(ohlcv_data.values())[0]
        else:
            # Already a DataFrame
            data_df = ohlcv_data
        
        # Step 1: Technical indicators
        indicators = TechnicalIndicators.calculate_indicators(data_df)
        
        # Step 2: Add sentiment from latest news
        if news_data:
            sentiments = []
            for item in news_data[-5:]:  # Last 5 news items
                result = self.sentiment_analyzer.analyze_sentiment(item['title'])
                sentiments.append(result['score'])
            
            # Average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            indicators['NEWS_SENTIMENT'] = avg_sentiment
        else:
            indicators['NEWS_SENTIMENT'] = 0.0
        
        # Step 3: Add price features
        if 'Close' in data_df.columns:
            close_prices = data_df['Close']
        elif isinstance(ohlcv_data, dict) and ticker in ohlcv_data:
            close_prices = ohlcv_data[ticker]['Close']
        else:
            # Fallback: try to get from indicators if price was already added
            close_prices = indicators.get('PRICE', pd.Series([100] * len(indicators), index=indicators.index))
        
        indicators['PRICE'] = close_prices
        indicators['PRICE_CHANGE'] = close_prices.pct_change()
        
        # Step 4: Normalize all features to [0, 1]
        # First normalize price features before calling normalize_indicators
        if 'PRICE' in indicators.columns:
            price_min = indicators['PRICE'].min()
            price_max = indicators['PRICE'].max()
            if price_max > price_min:
                indicators['PRICE'] = (indicators['PRICE'] - price_min) / (price_max - price_min)
            else:
                indicators['PRICE'] = 0.5
        
        if 'PRICE_CHANGE' in indicators.columns:
            indicators['PRICE_CHANGE'] = indicators['PRICE_CHANGE'].clip(-1.0, 1.0)
            indicators['PRICE_CHANGE'] = (indicators['PRICE_CHANGE'] + 1.0) / 2.0
        
        # Now normalize all other indicators
        normalized = self.indicators.normalize_indicators(indicators)
        
        print(f"✅ Created feature vector with {len(normalized.columns)} features")
        
        return normalized