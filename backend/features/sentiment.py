import json
import requests
from datetime import datetime

class LlamaSentimentAnalyzer:
    """
    Uses locally-running Llama 3.2 (via Ollama)
    Completely FREE, completely private, no API keys needed!
    """
    
    def __init__(self, model="llama3.2:1b", ollama_host="http://localhost:11434"):
        """
        Initialize sentiment analyzer with local Llama
        
        Args:
            model: llama3.2:1b or llama3.2:3b
            ollama_host: Where Ollama is running
        
        Note: Make sure 'ollama serve' is running first!
        """
        self.model = model
        self.ollama_host = ollama_host
        self.api_endpoint = f"{ollama_host}/api/generate"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama on {self.ollama_host}")
                print(f"   Using model: {self.model}")
            else:
                print(f"❌ Ollama not responding properly")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama on {self.ollama_host}")
            print(f"   Make sure to run: ollama serve")
            print(f"   Error: {e}")
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of financial news using LOCAL Llama 3.2
        
        Args:
            text: News headline or article text
        
        Returns:
            {
                'sentiment': 'bullish' | 'bearish' | 'neutral',
                'score': float (-1.0 to 1.0),
                'reasoning': str
            }
        """
        
        # Prompt engineered for financial sentiment
        prompt = f"""You are a financial sentiment analyst. Analyze this news for stock trading impact.

News: "{text}"

Respond in this EXACT format (important!):
SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
SCORE: [number from -1.0 to 1.0]
REASONING: [one line reason]

Remember: -1.0 = very bearish, 0 = neutral, 1.0 = very bullish"""
        
        try:
            # Call local Llama via Ollama
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Low temp for consistent results
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result['response']
                return self._parse_sentiment_response(output)
            else:
                print(f"❌ Error: {response.status_code}")
                return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': 'API Error'}
        
        except requests.exceptions.ConnectionError:
            print("❌ Cannot reach Ollama. Start it with: ollama serve")
            return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': 'Connection Error'}
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': 'Error'}
    
    def _parse_sentiment_response(self, response):
        """Parse Llama's response into structured format"""
        try:
            lines = response.strip().split('\n')
            sentiment = None
            score = 0.0
            reasoning = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('SENTIMENT:'):
                    sentiment = line.split(':')[1].strip().lower()
                elif line.startswith('SCORE:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
                    except:
                        score = 0.0
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':')[1].strip()
            
            return {
                'sentiment': sentiment or 'neutral',
                'score': score,
                'reasoning': reasoning
            }
        except Exception as e:
            print(f"Parse error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'reasoning': 'Parse error'}

# LEARNING EXERCISE
if __name__ == "__main__":
    print("🤖 Testing Local Llama 3.2 Sentiment Analysis\n")
    
    analyzer = LlamaSentimentAnalyzer(model="llama3.2:1b")
    
    # Test headlines
    test_headlines = [
        "Apple beats earnings expectations, stock rallies",
        "Tech stocks plunge amid recession fears",
        "Market remains stable as Fed pauses rate hikes",
        "GPU shortage expected to ease next quarter",
        "Company reports massive losses, shares tumble"
    ]
    
    print("📰 Analyzing financial headlines...\n")
    
    for headline in test_headlines:
        print(f"📰 '{headline}'")
        sentiment = analyzer.analyze_sentiment(headline)
        print(f"   ✅ Sentiment: {sentiment['sentiment'].upper()}")
        print(f"   📊 Score: {sentiment['score']:.2f}")
        print(f"   💡 Reason: {sentiment['reasoning']}\n")