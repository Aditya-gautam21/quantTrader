import feedparser
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class NewsCollector:
    def __init__(self):
        self.rss_feeds={
            'cnbc': 'https://feeds.cnbc.com/cnbc/financialnews/',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'nasdaq': 'https://feeds.nasdaq.com/nasdaq/news',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }

    def fetch_news(self, hours=6):
        cutoff_time = datetime.now - timedeta(hours=hours)
        all_news = []

        print(f"\n Fetching news form last {hours} hours!")
        
