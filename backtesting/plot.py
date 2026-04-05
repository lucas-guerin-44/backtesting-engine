"""Visualization module for backtest results.

Generates publication-ready charts with:
- Equity curve with drawdown shading
- Trade markers (entry/exit)
- Summary statistics annotation
- Multi-asset portfolio allocation comparison

Uses matplotlib. Install with: pip install matplotlib

Example::

    from backtesting.plot import plot_backtest, plot_portfolio
    plot_backtest(equity_curve, trades, title="Trend Following — XAUUSD H1")
    plot_portfolio({"Equal Weight": result_ew, "Risk Parity": result_rp})
"""

from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_backtest(
    equity_curve: np.ndarray,
    trades: list,
    timestamps: Optional[list] = None,
    title: str = "Backtest Results",
    starting_cash: float = 10_000,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot equity curve with drawdown shading and trade markers.

    Parameters
    ----------
    equity_curve : np.ndarray
        Array of equity values per bar.
    trades : list
        List of Trade objects (must have .entry_bar.ts, .pnl, .side).
    timestamps : list, optional
        Timestamps for x-axis. If None, uses bar index.
    title : str
        Chart title.
    starting_cash : float
        Starting cash for return calculation.
    save_path : str, optional
        If provided, saves the figure to this path.
    show : bool
        Whether to display the plot interactively.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    equity = np.asarray(equity_curve, dtype=np.float64)
    n = len(equity)
    x = timestamps if timestamps is not None else list(range(n))

    # Compute drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, 1.0)

    # Compute stats
    total_return = (equity[-1] - starting_cash) / starting_cash * 100
    max_dd = np.max(drawdown) * 100
    n_trades = len(trades)
    wins = len([t for t in trades if t.pnl and t.pnl > 0])
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    # --- Equity curve ---
    ax1.plot(x, equity, color="#2196F3", linewidth=1.2, label="Equity")
    ax1.axhline(y=starting_cash, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Trade markers
    for t in trades:
        # Find bar index for entry
        if timestamps is not None:
            try:
                idx = timestamps.index(t.entry_bar.ts)
            except (ValueError, AttributeError):
                continue
        else:
            continue

        if idx < n:
            color = "#4CAF50" if t.pnl and t.pnl > 0 else "#F44336"
            marker = "^" if t.side > 0 else "v"
            ax1.scatter(x[idx], equity[idx], color=color, marker=marker,
                       s=30, zorder=5, alpha=0.7)

    # Stats annotation
    stats_text = (
        f"Return: {total_return:+.2f}%\n"
        f"Max DD: {max_dd:.2f}%\n"
        f"Trades: {n_trades}\n"
        f"Win rate: {win_rate:.1f}%"
    )
    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax1.set_ylabel("Equity ($)")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Drawdown ---
    ax2.fill_between(x, 0, -drawdown * 100, color="#F44336", alpha=0.3)
    ax2.plot(x, -drawdown * 100, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time" if timestamps else "Bar")
    ax2.grid(True, alpha=0.3)

    # Format x-axis for dates
    if timestamps is not None and hasattr(timestamps[0], "date"):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_strategy_comparison(
    results: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot equity curves for multiple strategies on the same chart.

    Parameters
    ----------
    results : dict
        Mapping of strategy name -> equity curve (np.ndarray).
    save_path : str, optional
        If provided, saves the figure to this path.
    show : bool
        Whether to display the plot.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (name, equity) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ret = (equity[-1] - equity[0]) / equity[0] * 100
        ax.plot(equity, color=color, linewidth=1.2, label=f"{name} ({ret:+.1f}%)")

    ax.axhline(y=10_000, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Strategy Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_portfolio(
    results: dict,
    timestamps: Optional[list] = None,
    starting_cash: float = 100_000,
    title: str = "Multi-Asset Portfolio — Allocation Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot portfolio equity curves for multiple allocation schemes.

    Top panel: equity curves with return and max drawdown in the legend.
    Bottom panel: drawdown for each scheme.

    Parameters
    ----------
    results : dict
        Mapping of allocator name -> ``PortfolioBacktestResult`` (must have
        ``.equity_curve``) or plain ``np.ndarray``.
    timestamps : list, optional
        Timestamps for x-axis. If None, tries ``results[first].timestamps``,
        then falls back to bar index.
    starting_cash : float
        Starting cash for return calculation.
    title : str
        Chart title.
    save_path : str, optional
        If provided, saves the figure to this path.
    show : bool
        Whether to display the plot interactively.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#795548", "#607D8B"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.05})

    # Resolve timestamps from first result if not provided
    if timestamps is None:
        for v in results.values():
            if hasattr(v, "timestamps"):
                timestamps = v.timestamps
                break

    for i, (name, result) in enumerate(results.items()):
        equity = result.equity_curve if hasattr(result, "equity_curve") else np.asarray(result)
        equity = np.asarray(equity, dtype=np.float64)
        n = len(equity)
        x = timestamps[:n] if timestamps is not None else list(range(n))
        color = colors[i % len(colors)]

        ret = (equity[-1] - starting_cash) / starting_cash * 100
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
        max_dd = np.max(dd) * 100

        label = f"{name} ({ret:+.1f}%, DD {max_dd:.1f}%)"
        ax1.plot(x, equity, color=color, linewidth=1.4, label=label)
        ax2.plot(x, -dd * 100, color=color, linewidth=0.9, alpha=0.8)

    ax1.axhline(y=starting_cash, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Portfolio Equity ($)")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time" if timestamps else "Bar")
    ax2.grid(True, alpha=0.3)

    if timestamps is not None and len(timestamps) > 0 and hasattr(timestamps[0], "date"):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Portfolio chart saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
