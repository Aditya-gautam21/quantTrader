#!/usr/bin/env python3
"""
Test script for QuantTrader Real-Time API
"""
import requests
import time
import sys

API_URL = "http://localhost:8000"

def test_endpoint(name, url):
    """Test a single endpoint"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name}: OK")
            return True, data
        else:
            print(f"❌ {name}: Failed (Status {response.status_code})")
            return False, None
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: Connection refused (Is the server running?)")
        return False, None
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return False, None

def main():
    print("🧪 Testing QuantTrader API Endpoints\n")
    print(f"API URL: {API_URL}\n")
    
    # Test root
    success, data = test_endpoint("Root", f"{API_URL}/")
    if success:
        print(f"   Status: {data.get('status')}")
    print()
    
    # Test market data
    success, data = test_endpoint("Market Data", f"{API_URL}/api/market/BTC%2FUSDT")
    if success:
        symbol = data.get('symbol')
        data_points = len(data.get('data', []))
        print(f"   Symbol: {symbol}")
        print(f"   Data points: {data_points}")
        if data_points > 0:
            latest = data['data'][-1]
            print(f"   Latest Close: ${latest.get('Close', 0):,.2f}")
    print()
    
    # Test predictions
    success, data = test_endpoint("Predictions", f"{API_URL}/api/predictions/BTC%2FUSDT")
    if success:
        pred = data.get('prediction', {})
        print(f"   LSTM Signal: {pred.get('LSTM_Signal')}")
        print(f"   Transformer Signal: {pred.get('Transformer_Signal')}")
        print(f"   Confidence: {pred.get('Confidence')}%")
    print()
    
    # Test news
    success, data = test_endpoint("News Feed", f"{API_URL}/api/news")
    if success:
        news_count = len(data.get('news', []))
        print(f"   News items: {news_count}")
        if news_count > 0:
            print(f"   Latest: {data['news'][0].get('title', '')[:60]}...")
    print()
    
    print("=" * 60)
    print("✅ API Testing Complete!")

if __name__ == "__main__":
    main()
