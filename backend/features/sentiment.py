from typing import Dict, Any
from config import DEEPSEEK_MODEL
from llama_cpp import Llama
import pandas as pd
from transformers import pipeline
from transformers.utils import logging
import os

class LlamaSentimentAnalyzer:
    def sentiment_analyser(self, news):
        logging.set_verbosity_error()
        
        sentiment = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )

        print(f'\n {sentiment(news)}')


if __name__ == "__main__":
    print("Testing Local Llama Sentiment Analysis\n")

    analyzer = LlamaSentimentAnalyzer()

    test_headlines = []

    news = pd.read_csv("raw_data/2026-03-10/recent_crypto_news.csv")

    test_headlines.extend(news['title'].to_list())

    print("Analyzing financial headlines...\n")
    for headline in test_headlines:
        print(f"'{headline}'")
        result = analyzer.sentiment_analyser(headline)