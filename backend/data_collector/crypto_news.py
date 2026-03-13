import feedparser
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path


class NewsCollector:

    def __init__(self, data_dir="./raw_data"):

        self.rss_feeds = {
            "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
            "CryptoSlate": "https://cryptoslate.com/feed/",
            "Decrypt": "https://decrypt.co/feed",
            "Cointelegraph": "https://cointelegraph.com/rss",
            "BitcoinNews": "https://news.bitcoin.com/feed/"
        }

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_publish_time(self, entry):
        """
        Safely extract publish time from different RSS formats
        """

        if "published_parsed" in entry and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])

        if "updated_parsed" in entry and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])

        return None

    def fetch_news(self, hours=6):

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        all_news = []

        print("\n" + "=" * 60)
        print(f"Fetching crypto news from last {hours} hours")
        print("=" * 60)

        for source, url in self.rss_feeds.items():

            print(f"\nConnecting to {source}...")

            try:

                feed = feedparser.parse(url)

                if feed.bozo:
                    print(f"Warning: issue parsing {source}")

                source_count = 0

                print(f"Scanning {len(feed.entries)} articles...")

                for entry in feed.entries:

                    published = self.get_publish_time(entry)

                    if not published:
                        continue

                    if published > cutoff_time:

                        news_item = {
                            "source": source,
                            "title": entry.title,
                            "link": entry.link,
                            "published": published.isoformat(),
                            "summary": entry.get("summary", "")[:200]
                        }

                        all_news.append(news_item)

                        source_count += 1

                        print(f"✓ {entry.title[:80]}")

                print(f"{source} → {source_count} recent articles")

            except Exception as e:

                print(f"Error fetching {source}: {e}")

        print("\nTotal news collected:", len(all_news))

        return all_news

    def save_news(self, news_items):

        current_date = str(datetime.now().date())

        save_dir = self.data_dir / current_date
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = save_dir / "recent_crypto_news.csv"

        df = pd.DataFrame(news_items)

        df.to_csv(filename, index=False)

        print(f"\nSaved {len(df)} articles to {filename}")

        return df


def run_live_collector(interval_minutes=10, hours=6):

    collector = NewsCollector()

    print("\nStarting LIVE crypto news monitor")
    print(f"Update interval: {interval_minutes} minutes\n")

    while True:

        print("\n" + "#" * 60)
        print("RUN TIME:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("#" * 60)

        news = collector.fetch_news(hours)

        if news:
            collector.save_news(news)

        else:
            print("No recent news found")

        print(f"\nSleeping for {interval_minutes} minutes...\n")

        time.sleep(interval_minutes * 60)


if __name__ == "__main__":

    # run once
    collector = NewsCollector()

    news = collector.fetch_news(hours=6)

    if news:
        collector.save_news(news)

    print("\nTop headlines:\n")

    for i, item in enumerate(news[:5]):

        print(f"{i+1}. [{item['source']}] {item['title']}")
        print(item["link"])
        print()

        
    #Run continuously at interval of 10 mins
    #run_live_collector(interval_minutes=10)