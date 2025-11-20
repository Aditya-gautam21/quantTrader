"""
Model configuration for LLM trading projects
"""
import os

# Try environment variables first (if set), fallback to relative paths
BASE_DIR = os.path.expanduser("~/llama.cpp")

DEEPSEEK_MODEL = os.environ.get(
    "DEEPSEEK_MODEL",
    os.path.join(BASE_DIR, "models/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf")
)

LLAMA_MODEL = os.environ.get(
    "LLAMA_MODEL", 
    os.path.join(BASE_DIR, "models/Llama-3.2-3B-Instruct-f16.gguf")
)

# GPU settings for your GTX 1650 Ti
DEEPSEEK_GPU_LAYERS = 18  # Optimal for 8B Q5_K_M
LLAMA_GPU_LAYERS = 25      # Lighter 3B model can use more

# Default parameters
DEFAULT_CTX = 4096
DEFAULT_THREADS = 6
DEFAULT_TEMP = 0.3  # Lower for trading (more consistent)

# Validate models exist
def validate_models():
    """Check if model files exist"""
    if not os.path.exists(DEEPSEEK_MODEL):
        raise FileNotFoundError(f"DeepSeek model not found: {DEEPSEEK_MODEL}")
    if not os.path.exists(LLAMA_MODEL):
        raise FileNotFoundError(f"Llama model not found: {LLAMA_MODEL}")
    print("✓ All models found")
    return True

if __name__ == "__main__":
    print(f"DeepSeek: {DEEPSEEK_MODEL}")
    print(f"Llama 3.2: {LLAMA_MODEL}")
    validate_models()
