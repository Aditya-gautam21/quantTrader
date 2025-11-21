import feedparser
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class NewsCollector:
    def __init__(self, data_dir="./raw_data"):
        self.rss_feeds={
            'cnbc': 'https://feeds.cnbc.com/cnbc/financialnews/',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'nasdaq': 'https://feeds.nasdaq.com/nasdaq/news',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_news(self, hours=6):
        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_news = []

        print(f"\n Fetching news form last {hours} hours!")

        for source, url in self.rss_feeds.items():
            try:
                print(f"\n Fetching from {source}...")
                feed = feedparser.parse(url)

                for entry in feed.entries[:10]:
                    try:
                        published = datetime(*entry.published_parsed[:6])

                        if published > cutoff_time:
                            all_news.append({
                                'source': source,
                                'title': entry.title,
                                'link': entry.link,
                                'published': published.isoformat(),
                                'summary': entry.get('summary', '')[:200]
                            })
                    except:
                        pass
                
                print(f"\n Fetched {len(all_news)} news items from {source}!")
            
            except Exception as e:
                print(f"Error fetching from {source}: {e}")
            
        return all_news
    
    def save_news(self, news_items, filename=f"recent_news.csv"):
        current_data_str = str(datetime.now().date())

        ticker_dir = self.data_dir / current_data_str
        ticker_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(news_items)
        filepath = ticker_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\n Saved news to {filepath}, Total items: {len(df)}")
        return df
    
if __name__ == '__main__':
    collector = NewsCollector()

    news = collector.fetch_news(hours=6)

    print(f"Fetched total {len(news)} news items.")

    for i, item in enumerate(news[:3]):
        print(f"{i+1}. [{item['source'].upper()}] {item['title'][:70]}...")
        print(f"   Link: {item['link']}")
        print()

        collector.save_news(news)