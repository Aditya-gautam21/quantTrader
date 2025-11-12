from typing import Dict, Any

try:
    from llama_cpp import Llama
except ImportError as e:
    raise SystemExit(
        "llama-cpp-python is not installed.\n"
        "Install CPU:   pip install llama-cpp-python\n"
    ) from e


class LlamaSentimentAnalyzer:
    """
    Uses locally-running Llama via llama.cpp (llama-cpp-python).
    Completely FREE, completely offline, no API keys.
    """

    def __init__(
        self,
        model_path: str = r"C:\Users\ASUS\models\Llama-3.2-3B-Instruct-f16.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: Full path to GGUF file.
            n_ctx: Context window tokens.
            n_threads: CPU threads.
            n_gpu_layers: GPU layers (0 = CPU only).
            verbose: Print model info.
        """
        print(f"Loading model: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        
        print("Model loaded successfully")

    def _build_prompt(self, text: str) -> str:
        return f"""You are a financial sentiment analyst. Analyze this news for stock trading impact.

News: "{text}"

Respond in this EXACT format:
SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
SCORE: [number from -1.0 to 1.0]
REASONING: [one line reason]

Remember: -1.0 = very bearish, 0 = neutral, 1.0 = very bullish"""

    def _parse_sentiment_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Llama's text into a structured result."""
        try:
            lines = response_text.strip().splitlines()
            sentiment = "neutral"
            score = 0.0
            reasoning = ""

            for line in lines:
                s = line.strip()
                if s.upper().startswith("SENTIMENT:"):
                    sentiment = s.split(":", 1)[1].strip().lower()
                elif s.upper().startswith("SCORE:"):
                    try:
                        score = float(s.split(":", 1)[1].strip())
                        score = max(-1.0, min(1.0, score))
                    except Exception:
                        score = 0.0
                elif s.upper().startswith("REASONING:"):
                    reasoning = s.split(":", 1)[1].strip()

            return {
                "sentiment": sentiment,
                "score": score,
                "reasoning": reasoning,
            }
        except Exception as e:
            print(f"Parse error: {e}")
            return {"sentiment": "neutral", "score": 0.0, "reasoning": "Parse error"}

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of financial news using local llama.cpp.
        """
        prompt = self._build_prompt(text)

        try:
            out = self.llm.create_completion(
                prompt=prompt,
                temperature=0.3,
                top_p=0.9,
                max_tokens=256,
                stop=None,
            )

            if "choices" in out and out["choices"]:
                model_text = out["choices"][0].get("text", "").strip()
            else:
                model_text = ""

            return self._parse_sentiment_response(model_text)
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "score": 0.0, "reasoning": "Analysis error"}


if __name__ == "__main__":
    print("Testing Local Llama Sentiment Analysis\n")

    analyzer = LlamaSentimentAnalyzer()

    test_headlines = [
        "Apple beats earnings expectations, stock rallies",
        "Tech stocks plunge amid recession fears",
        "Market remains stable as Fed pauses rate hikes",
        "GPU shortage expected to ease next quarter",
        "Company reports massive losses, shares tumble",
    ]

    print("Analyzing financial headlines...\n")
    for headline in test_headlines:
        print(f"'{headline}'")
        result = analyzer.analyze_sentiment(headline)
        print(f"   Sentiment: {result['sentiment'].upper()}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Reason: {result['reasoning']}\n")