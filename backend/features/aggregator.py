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
        self.sentiment_analyzer = LlamaSentimentAnalyzer(model="llama3.2:1b")
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
        
        # Step 1: Technical indicators
        indicators = self.indicators.calculate_indicators(ohlcv_data, ticker=ticker)
        
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
        close_prices = ohlcv_data[(ticker, 'Close')]
        indicators['PRICE'] = close_prices
        indicators['PRICE_CHANGE'] = close_prices.pct_change()
        
        # Step 4: Normalize all features to [0, 1]
        normalized = self.indicators.normalize_indicators(indicators)
        
        print(f"✅ Created feature vector with {len(normalized.columns)} features")
        
        return normalized