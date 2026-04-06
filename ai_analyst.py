"""AI-powered post-backtest analyst using a local LLM via Ollama.

Feeds backtest metrics into a small local model and prints a plain-English
analysis covering performance, risk, and potential improvements.

Requires Ollama running locally (https://ollama.com).
Enable via AI_ANALYST=true in .env.

Usage::

    from ai_analyst import analyze_backtest, analyze_portfolio

    # After a single-asset backtest
    analyze_backtest("Trend Following", metrics)

    # After a portfolio backtest
    analyze_portfolio(portfolio_metrics)
"""

import json
import sys
from typing import Any, Dict, List, Optional

# Windows consoles may use cp1252 which can't print all unicode.
# Reconfigure stdout once to handle arbitrary LLM output.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

import requests

from config import AI_ANALYST, AI_MODEL, OLLAMA_URL

_SYSTEM_PROMPT = """\
You are a quantitative trading analyst reviewing backtest results.
Be direct, specific, and opinionated. Reference the actual numbers.
Structure your response as:

1. **Performance verdict** (1-2 sentences: is this strategy worth trading?)
2. **Risk assessment** (drawdown, win rate, trade count — are they acceptable?)
3. **Red flags** (overfitting signals, too few trades, suspiciously good numbers, etc.)
4. **Next steps** — give 2-3 specific, actionable things to try next. Be concrete:
   name parameter ranges to test, strategies to swap in, timeframes to try,
   filters to add (e.g. "add a volatility filter to skip entries when ATR > 2x
   its 50-bar average"). Think like a senior quant telling a junior what to run
   next, not generic advice.

Keep it under 200 words. No fluff, no disclaimers.\
"""

_PORTFOLIO_SYSTEM_PROMPT = """\
You are a quantitative portfolio analyst reviewing multi-asset backtest results.
Be direct, specific, and opinionated. Reference the actual numbers.
Structure your response as:

1. **Portfolio verdict** (1-2 sentences: is this portfolio construction sound?)
2. **Allocation assessment** (which allocator performed best and why)
3. **Red flags** (concentration risk, correlated drawdowns, overfitting, etc.)
4. **Next steps** — give 2-3 specific, actionable things to try next. Be concrete:
   which assets to drop or add, allocation weights to override, rebalance
   frequencies to test, correlation thresholds to tune. Think like a portfolio
   manager telling the team what to run in the next iteration.

Keep it under 200 words. No fluff, no disclaimers.\
"""


def is_available() -> bool:
    """Check if Ollama is running and the configured model is available."""
    if not AI_ANALYST:
        return False
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Match with or without :latest tag
        return any(AI_MODEL in m for m in models)
    except (requests.ConnectionError, requests.Timeout, Exception):
        return False


def _pull_model() -> bool:
    """Pull the configured model if not already available."""
    print(f"  Pulling {AI_MODEL} (first run only)...")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"model": AI_MODEL, "stream": True},
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "pulling" in status and "completed" not in status:
                    # Show download progress
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    if total > 0:
                        pct = completed / total * 100
                        print(f"\r  Downloading: {pct:.0f}%", end="", flush=True)
                elif status == "success":
                    print(f"\r  Download complete.          ")
                    return True
        return True
    except Exception as e:
        print(f"  Failed to pull model: {e}")
        return False


def _ensure_model() -> bool:
    """Ensure the model is available, pulling it if necessary."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if any(AI_MODEL in m for m in models):
            return True
        return _pull_model()
    except (requests.ConnectionError, requests.Timeout):
        print(f"  Ollama not running at {OLLAMA_URL}. Install from https://ollama.com")
        return False


def _chat(system_prompt: str, user_message: str) -> Optional[str]:
    """Send a chat request to Ollama and return the response text."""
    if not _ensure_model():
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": AI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except requests.ConnectionError:
        print(f"  Ollama not running at {OLLAMA_URL}. Install from https://ollama.com")
        return None
    except Exception as e:
        print(f"  AI analyst error: {e}")
        return None


def analyze_backtest(results: Dict[str, Dict[str, Any]]) -> None:
    """Analyze one or more single-asset backtest results.

    Parameters
    ----------
    results : dict
        Mapping of strategy name to metrics dict. Each metrics dict should have:
        - pct_return, sharpe, max_drawdown, total_trades, win_rate
        - Optionally: optimization and walk_forward sub-dicts
    """
    if not AI_ANALYST:
        return

    lines = []
    for name, m in results.items():
        lines.append(f"Strategy: {name}")
        lines.append(f"  Return: {m.get('pct_return', 0):+.2f}%")
        lines.append(f"  Sharpe: {m.get('sharpe', 0):.4f}")
        lines.append(f"  Max Drawdown: {m.get('max_drawdown', 0):.2f}%")
        lines.append(f"  Trades: {m.get('total_trades', 0)}")
        lines.append(f"  Win Rate: {m.get('win_rate', 0):.1f}%")

        if "optimization" in m:
            opt = m["optimization"]
            lines.append(f"  Optimization: best Sharpe {opt.get('best_sharpe', 0):.4f} "
                         f"({opt.get('n_trials', 0)} trials)")

        if "walk_forward" in m:
            wf = m["walk_forward"]
            lines.append(f"  Walk-Forward: IS={wf.get('is_mean', 0):.4f} "
                         f"OOS={wf.get('oos_mean', 0):.4f} "
                         f"degradation={wf.get('degradation', 0):.4f}")
        lines.append("")

    prompt = "\n".join(lines)
    response = _chat(_SYSTEM_PROMPT, prompt)
    if response:
        print()
        print("=" * 60)
        print("  AI Analyst")
        print("=" * 60)
        print()
        print(response)
        print()


def analyze_portfolio(
    allocator_results: Dict[str, Dict[str, Any]],
    walk_forward: Optional[Dict[str, Any]] = None,
) -> None:
    """Analyze portfolio backtest results across allocation schemes.

    Parameters
    ----------
    allocator_results : dict
        Mapping of allocator name to metrics dict. Each should have:
        - pct_return, sharpe, max_drawdown, total_trades
        - Optionally: weights (dict of symbol -> weight)
    walk_forward : dict, optional
        Walk-forward results with is_mean, oos_mean, degradation.
    """
    if not AI_ANALYST:
        return

    lines = []
    for name, m in allocator_results.items():
        lines.append(f"Allocator: {name}")
        lines.append(f"  Return: {m.get('pct_return', 0):+.2f}%")
        lines.append(f"  Sharpe: {m.get('sharpe', 0):.4f}")
        lines.append(f"  Max Drawdown: {m.get('max_drawdown', 0):.2f}%")
        lines.append(f"  Trades: {m.get('total_trades', 0)}")
        if "weights" in m:
            weights_str = ", ".join(f"{s}: {w:.1%}" for s, w in m["weights"].items())
            lines.append(f"  Final Weights: {weights_str}")
        lines.append("")

    if walk_forward:
        lines.append("Walk-Forward Validation:")
        lines.append(f"  In-Sample Mean: {walk_forward.get('is_mean', 0):.4f}")
        lines.append(f"  Out-of-Sample Mean: {walk_forward.get('oos_mean', 0):.4f}")
        lines.append(f"  Degradation: {walk_forward.get('degradation', 0):.4f}")

    prompt = "\n".join(lines)
    response = _chat(_PORTFOLIO_SYSTEM_PROMPT, prompt)
    if response:
        print()
        print("=" * 70)
        print("  AI Analyst — Portfolio Review")
        print("=" * 70)
        print()
        print(response)
        print()
