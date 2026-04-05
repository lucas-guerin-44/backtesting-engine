"""SQLite-backed results database for backtests and optimizations.

Persists backtest runs, optimization results, and walk-forward splits
so they can be queried later. Zero external dependencies (SQLite is stdlib).

Usage::

    from results_db import ResultsDB

    with ResultsDB("results.db") as db:
        run_id = db.save_run("TrendFollowing", params, equity_curve, trades)
        print(db.query_runs(min_sharpe=0.3))

CLI::

    python -m results_db query --min-sharpe 0.3
    python -m results_db get 42
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import compute_sharpe


@dataclass
class RunRecord:
    """A single backtest run from the database."""
    id: int
    timestamp: str
    strategy_name: str
    params: Dict[str, Any]
    objective: str
    starting_cash: float
    commission_bps: float
    slippage_bps: float
    data_hash: str
    final_equity: float
    pct_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    win_count: int
    loss_count: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    strategy_name TEXT NOT NULL,
    params_json TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT 'sharpe',
    starting_cash REAL NOT NULL,
    commission_bps REAL NOT NULL DEFAULT 0.0,
    slippage_bps REAL NOT NULL DEFAULT 0.0,
    data_hash TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    final_equity REAL,
    pct_return REAL,
    sharpe REAL,
    max_drawdown REAL,
    total_trades INTEGER,
    win_rate REAL,
    win_count INTEGER,
    loss_count INTEGER
);

CREATE TABLE IF NOT EXISTS walk_forward_splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wf_run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    split_idx INTEGER NOT NULL,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    is_score REAL,
    oos_score REAL,
    oos_return REAL,
    oos_max_dd REAL,
    best_params_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_name);
CREATE INDEX IF NOT EXISTS idx_results_sharpe ON results(sharpe);
CREATE INDEX IF NOT EXISTS idx_wf_splits_run ON walk_forward_splits(wf_run_id);
"""


class ResultsDB:
    """SQLite-backed results database.

    Parameters
    ----------
    path : str or Path
        Path to the SQLite database file. Created if it does not exist.
    """

    def __init__(self, path: str = "results.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- Save methods --

    def save_run(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        equity_curve: np.ndarray,
        trades: list,
        objective: str = "sharpe",
        starting_cash: float = 10_000,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        data_hash: str = "",
        freq_per_year: int = 252,
    ) -> int:
        """Save a single backtest run. Returns the run_id."""
        equity_curve = np.asarray(equity_curve, dtype=np.float64)

        # Compute metrics
        final_eq = float(equity_curve[-1]) if len(equity_curve) > 0 else starting_cash
        pct_return = (final_eq - starting_cash) / starting_cash * 100
        sharpe = compute_sharpe(equity_curve, freq_per_year=freq_per_year)

        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        wins = [t for t in trades if getattr(t, "pnl", None) and t.pnl > 0]
        losses = [t for t in trades if getattr(t, "pnl", None) and t.pnl <= 0]
        total = len(wins) + len(losses)
        win_rate = len(wins) / total * 100 if total > 0 else 0.0

        # Serialize params (convert numpy types to Python types)
        clean_params = {k: _to_python(v) for k, v in params.items()}

        cur = self.conn.execute(
            "INSERT INTO runs (strategy_name, params_json, objective, starting_cash, "
            "commission_bps, slippage_bps, data_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (strategy_name, json.dumps(clean_params), objective, starting_cash,
             commission_bps, slippage_bps, data_hash),
        )
        run_id = cur.lastrowid

        self.conn.execute(
            "INSERT INTO results (run_id, final_equity, pct_return, sharpe, max_drawdown, "
            "total_trades, win_rate, win_count, loss_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, final_eq, pct_return, sharpe, max_dd, total, win_rate,
             len(wins), len(losses)),
        )
        self.conn.commit()
        return run_id

    def save_walk_forward(
        self,
        wf_result,
        strategy_name: str,
        objective: str = "sharpe",
        starting_cash: float = 10_000,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        data_hash: str = "",
    ) -> int:
        """Save all walk-forward splits. Returns the parent wf_run_id."""
        params = {
            "type": "walk_forward",
            "n_splits": len(wf_result.splits),
            "in_sample_mean": wf_result.in_sample_mean,
            "out_of_sample_mean": wf_result.out_of_sample_mean,
            "degradation": wf_result.degradation,
        }

        cur = self.conn.execute(
            "INSERT INTO runs (strategy_name, params_json, objective, starting_cash, "
            "commission_bps, slippage_bps, data_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (strategy_name, json.dumps(params), f"walk_forward:{objective}",
             starting_cash, commission_bps, slippage_bps, data_hash),
        )
        wf_run_id = cur.lastrowid

        # Store aggregate metrics in results
        self.conn.execute(
            "INSERT INTO results (run_id, sharpe, pct_return) VALUES (?, ?, ?)",
            (wf_run_id, wf_result.out_of_sample_mean,
             np.mean([s.get("oos_return_pct", 0) for s in wf_result.splits])),
        )

        # Store individual splits
        for s in wf_result.splits:
            best_params = s.get("best_params", {})
            clean_params = {k: _to_python(v) for k, v in best_params.items()}
            self.conn.execute(
                "INSERT INTO walk_forward_splits (wf_run_id, split_idx, train_start, "
                "train_end, test_start, test_end, is_score, oos_score, oos_return, "
                "oos_max_dd, best_params_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (wf_run_id, s["split"], s.get("train_start"), s.get("train_end"),
                 s.get("test_start"), s.get("test_end"),
                 s.get("in_sample_score"), s.get("out_of_sample_score"),
                 s.get("oos_return_pct"), s.get("oos_max_dd_pct"),
                 json.dumps(clean_params)),
            )

        self.conn.commit()
        return wf_run_id

    # -- Query methods --

    def query_runs(
        self,
        strategy: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None,
        limit: int = 50,
        order_by: str = "sharpe DESC",
    ) -> pd.DataFrame:
        """Query runs with optional filters. Returns DataFrame."""
        conditions = []
        params = []

        if strategy:
            conditions.append("r.strategy_name = ?")
            params.append(strategy)
        if min_sharpe is not None:
            conditions.append("res.sharpe >= ?")
            params.append(min_sharpe)
        if max_drawdown is not None:
            conditions.append("res.max_drawdown <= ?")
            params.append(max_drawdown)
        if min_trades is not None:
            conditions.append("res.total_trades >= ?")
            params.append(min_trades)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Validate order_by to prevent SQL injection
        allowed_cols = {"sharpe", "pct_return", "max_drawdown", "total_trades",
                        "win_rate", "final_equity", "r.timestamp"}
        order_col = order_by.split()[0].lower()
        if order_col not in allowed_cols:
            order_by = "sharpe DESC"

        query = f"""
            SELECT r.id, r.timestamp, r.strategy_name, r.params_json, r.objective,
                   res.final_equity, res.pct_return, res.sharpe, res.max_drawdown,
                   res.total_trades, res.win_rate, res.win_count, res.loss_count
            FROM runs r
            LEFT JOIN results res ON r.id = res.run_id
            {where}
            ORDER BY res.{order_by}
            LIMIT ?
        """
        params.append(limit)

        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def get_run(self, run_id: int) -> Optional[RunRecord]:
        """Get full details of a single run."""
        cur = self.conn.execute("""
            SELECT r.id, r.timestamp, r.strategy_name, r.params_json, r.objective,
                   r.starting_cash, r.commission_bps, r.slippage_bps, r.data_hash,
                   res.final_equity, res.pct_return, res.sharpe, res.max_drawdown,
                   res.total_trades, res.win_rate, res.win_count, res.loss_count
            FROM runs r
            LEFT JOIN results res ON r.id = res.run_id
            WHERE r.id = ?
        """, (run_id,))
        row = cur.fetchone()
        if row is None:
            return None

        return RunRecord(
            id=row[0], timestamp=row[1], strategy_name=row[2],
            params=json.loads(row[3]), objective=row[4],
            starting_cash=row[5], commission_bps=row[6],
            slippage_bps=row[7], data_hash=row[8],
            final_equity=row[9] or 0, pct_return=row[10] or 0,
            sharpe=row[11] or 0, max_drawdown=row[12] or 0,
            total_trades=row[13] or 0, win_rate=row[14] or 0,
            win_count=row[15] or 0, loss_count=row[16] or 0,
        )

    def get_walk_forward_splits(self, wf_run_id: int) -> pd.DataFrame:
        """Get all splits for a walk-forward run."""
        return pd.read_sql_query(
            "SELECT * FROM walk_forward_splits WHERE wf_run_id = ? ORDER BY split_idx",
            self.conn, params=(wf_run_id,),
        )

    def delete_run(self, run_id: int) -> bool:
        """Delete a run and its associated data (cascades)."""
        cur = self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- Utility --

    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """Hash DataFrame shape + head/tail for reproducibility tracking."""
        h = hashlib.sha256()
        h.update(str(df.shape).encode())
        h.update(str(df.columns.tolist()).encode())
        if len(df) > 0:
            h.update(df.head(5).to_csv().encode())
            h.update(df.tail(5).to_csv().encode())
        return h.hexdigest()[:16]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _to_python(v):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query backtesting results database")
    sub = parser.add_subparsers(dest="command")

    q = sub.add_parser("query", help="Search runs")
    q.add_argument("--strategy", type=str, help="Filter by strategy name")
    q.add_argument("--min-sharpe", type=float, help="Minimum Sharpe ratio")
    q.add_argument("--max-drawdown", type=float, help="Maximum drawdown")
    q.add_argument("--min-trades", type=int, help="Minimum number of trades")
    q.add_argument("--limit", type=int, default=20, help="Max results")

    g = sub.add_parser("get", help="Get run details")
    g.add_argument("run_id", type=int, help="Run ID")

    d = sub.add_parser("splits", help="Get walk-forward splits")
    d.add_argument("wf_run_id", type=int, help="Walk-forward run ID")

    args = parser.parse_args()
    db = ResultsDB()

    if args.command == "query":
        df = db.query_runs(
            strategy=args.strategy, min_sharpe=args.min_sharpe,
            max_drawdown=args.max_drawdown, min_trades=args.min_trades,
            limit=args.limit,
        )
        if df.empty:
            print("No runs found.")
        else:
            print(df.to_string(index=False))
    elif args.command == "get":
        run = db.get_run(args.run_id)
        if run:
            for field, val in run.__dict__.items():
                print(f"  {field}: {val}")
        else:
            print(f"Run {args.run_id} not found.")
    elif args.command == "splits":
        df = db.get_walk_forward_splits(args.wf_run_id)
        if df.empty:
            print("No splits found.")
        else:
            print(df.to_string(index=False))
    else:
        parser.print_help()

    db.close()
