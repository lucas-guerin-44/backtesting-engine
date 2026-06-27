"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# No localhost fallback: a missing DATALAKE_URL must fail loudly rather than
# silently point a process at a dead local port. The research repo owns the data
# layer now (see ../quant-strategies-research/data/); this is only for the engine's
# own standalone app (api.py / frontend.py / examples).
DATALAKE_URL = os.getenv("DATALAKE_URL", "").strip()
DATALAKE_API_KEY = os.getenv("DATALAKE_API_KEY", "").strip()
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://127.0.0.1:8001").strip()
LOCAL_API_PORT = os.getenv("LOCAL_API_PORT", "8001").strip()

# AI Analyst (requires Ollama: https://ollama.com)
AI_ANALYST = os.getenv("AI_ANALYST", "false").strip().lower() in ("true", "1", "yes")
AI_MODEL = os.getenv("AI_MODEL", "llama3.2:3b").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip()
