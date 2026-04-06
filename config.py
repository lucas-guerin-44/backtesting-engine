"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATALAKE_URL = os.getenv("DATALAKE_URL", "http://127.0.0.1:8008").strip()
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://127.0.0.1:8001").strip()
LOCAL_API_PORT = os.getenv("LOCAL_API_PORT", "8001").strip()

# AI Analyst (requires Ollama: https://ollama.com)
AI_ANALYST = os.getenv("AI_ANALYST", "false").strip().lower() in ("true", "1", "yes")
AI_MODEL = os.getenv("AI_MODEL", "llama3.2:3b").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip()
