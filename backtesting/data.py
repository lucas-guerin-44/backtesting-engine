"""OHLC data validation.

Catches data quality issues before they silently corrupt backtest results.
Returns a structured report with errors (must fix) and warnings (review).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class Severity(Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single data quality issue."""
    severity: Severity
    field: str
    message: str
    row_indices: Optional[List[int]] = None


@dataclass
class ValidationReport:
    """Result of OHLC data validation."""
    issues: List[ValidationIssue] = field(default_factory=list)
    n_bars: int = 0
    n_errors: int = 0
    n_warnings: int = 0

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings are OK)."""
        return self.n_errors == 0

    def raise_if_invalid(self) -> None:
        """Raise ValueError listing all errors if validation failed."""
        if self.is_valid:
            return
        errors = [iss for iss in self.issues if iss.severity == Severity.ERROR]
        msgs = [f"  - [{iss.field}] {iss.message}" for iss in errors]
        raise ValueError(
            f"OHLC data validation failed with {len(errors)} error(s):\n" + "\n".join(msgs)
        )

    def __str__(self) -> str:
        if not self.issues:
            return f"Validation passed: {self.n_bars} bars, no issues."
        lines = [f"Validation: {self.n_bars} bars, {self.n_errors} error(s), {self.n_warnings} warning(s)"]
        for iss in self.issues:
            tag = "ERROR" if iss.severity == Severity.ERROR else "WARN"
            n_rows = f" ({len(iss.row_indices)} rows)" if iss.row_indices else ""
            lines.append(f"  [{tag}] {iss.field}: {iss.message}{n_rows}")
        return "\n".join(lines)


def validate_ohlc(
    df: pd.DataFrame,
    expected_freq: Optional[str] = None,
    max_gap_multiple: int = 3,
) -> ValidationReport:
    """Validate OHLC DataFrame for common data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC data. Must have columns ``open``, ``high``, ``low``, ``close``
        and either a DatetimeIndex or a ``timestamp`` column.
    expected_freq : str, optional
        Expected bar frequency (e.g., ``"1h"``, ``"5min"``). If provided,
        detects gaps exceeding ``max_gap_multiple`` of this frequency.
        If omitted, frequency is inferred from the median time delta.
    max_gap_multiple : int
        Gaps larger than this multiple of expected frequency are flagged.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # --- Resolve index ---
    if isinstance(df.index, pd.DatetimeIndex):
        ts_index = df.index
    elif "timestamp" in df.columns:
        ts_index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    else:
        ts_index = None

    # 1. Required columns
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "columns", f"Missing required columns: {missing}"))
        report.n_errors += 1
        report.n_bars = len(df)
        return report  # Can't proceed without OHLC columns

    report.n_bars = len(df)
    if report.n_bars == 0:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "data", "DataFrame is empty"))
        report.n_errors += 1
        return report

    ohlc = df[["open", "high", "low", "close"]]

    # 2. Numeric types
    non_numeric = [col for col in ohlc.columns if not np.issubdtype(ohlc[col].dtype, np.number)]
    if non_numeric:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "dtype", f"Non-numeric OHLC columns: {non_numeric}"))
        report.n_errors += 1
        return report

    # 3. NaN / inf
    nan_mask = ohlc.isna().any(axis=1)
    if nan_mask.any():
        indices = list(np.where(nan_mask)[0][:20])  # Cap at 20 for readability
        report.issues.append(ValidationIssue(
            Severity.ERROR, "nan", f"NaN values in OHLC data", row_indices=indices))
        report.n_errors += 1

    inf_mask = np.isinf(ohlc.to_numpy()).any(axis=1)
    if inf_mask.any():
        indices = list(np.where(inf_mask)[0][:20])
        report.issues.append(ValidationIssue(
            Severity.ERROR, "inf", f"Infinite values in OHLC data", row_indices=indices))
        report.n_errors += 1

    # 4. OHLC consistency
    h, l, o_col, c_col = ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), ohlc["open"].to_numpy(), ohlc["close"].to_numpy()

    hl_bad = np.where(h < l)[0]
    if len(hl_bad) > 0:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "high_low", f"high < low on {len(hl_bad)} bars",
            row_indices=list(hl_bad[:20])))
        report.n_errors += 1

    ho_bad = np.where(h < o_col)[0]
    hc_bad = np.where(h < c_col)[0]
    high_bad = np.union1d(ho_bad, hc_bad)
    if len(high_bad) > 0:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "high", f"high < open or high < close on {len(high_bad)} bars",
            row_indices=list(high_bad[:20])))
        report.n_errors += 1

    lo_bad = np.where(l > o_col)[0]
    lc_bad = np.where(l > c_col)[0]
    low_bad = np.union1d(lo_bad, lc_bad)
    if len(low_bad) > 0:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "low", f"low > open or low > close on {len(low_bad)} bars",
            row_indices=list(low_bad[:20])))
        report.n_errors += 1

    # 5-7: Timestamp checks (only if we have timestamps)
    if ts_index is not None and len(ts_index) > 1:
        # 5. Duplicates
        dup_mask = ts_index.duplicated()
        if dup_mask.any():
            indices = list(np.where(dup_mask)[0][:20])
            report.issues.append(ValidationIssue(
                Severity.ERROR, "timestamp", f"{dup_mask.sum()} duplicate timestamps",
                row_indices=indices))
            report.n_errors += 1

        # 6. Monotonicity
        if not ts_index.is_monotonic_increasing:
            diffs = ts_index[1:] - ts_index[:-1]
            non_mono = np.where(diffs < pd.Timedelta(0))[0]
            report.issues.append(ValidationIssue(
                Severity.ERROR, "timestamp", f"Non-monotonic timestamps at {len(non_mono)} points",
                row_indices=list(non_mono[:20])))
            report.n_errors += 1

        # 7. Gap detection
        if ts_index.is_monotonic_increasing and len(ts_index) > 2:
            deltas = ts_index[1:] - ts_index[:-1]
            if expected_freq is not None:
                freq_td = pd.Timedelta(expected_freq)
            else:
                freq_td = deltas.median()

            if freq_td > pd.Timedelta(0):
                threshold = freq_td * max_gap_multiple
                gap_mask = deltas > threshold
                if gap_mask.any():
                    gap_indices = list(np.where(gap_mask)[0][:20])
                    report.issues.append(ValidationIssue(
                        Severity.WARNING, "gaps",
                        f"{gap_mask.sum()} gaps exceeding {max_gap_multiple}x expected frequency ({freq_td})",
                        row_indices=gap_indices))
                    report.n_warnings += 1

    # 8. Zero/negative prices
    nonpos = (ohlc <= 0).any(axis=1)
    if nonpos.any():
        indices = list(np.where(nonpos)[0][:20])
        report.issues.append(ValidationIssue(
            Severity.WARNING, "price", f"{nonpos.sum()} bars with zero or negative prices",
            row_indices=indices))
        report.n_warnings += 1

    # 9. Extreme moves
    if len(c_col) > 1:
        returns = np.abs(np.diff(c_col) / c_col[:-1])
        extreme = np.where(returns > 0.5)[0]
        if len(extreme) > 0:
            report.issues.append(ValidationIssue(
                Severity.WARNING, "extreme_move",
                f"{len(extreme)} bars with >50% single-bar move (possible bad data)",
                row_indices=list(extreme[:20])))
            report.n_warnings += 1

    return report
