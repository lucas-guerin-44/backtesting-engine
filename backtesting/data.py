"""OHLC data validation.

Catches data quality issues before they silently corrupt backtest results.
Returns a structured report with errors (must fix) and warnings (review).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


class Severity(Enum):
    WARNING = "warning"
    ERROR = "error"


#: Default severity per check. Corruption classes are ERROR by default; the
#: caller may override any of these via :class:`ValidationConfig`.
DEFAULT_SEVERITIES: Dict[str, Severity] = {
    "columns": Severity.ERROR,
    "data": Severity.ERROR,
    "dtype": Severity.ERROR,
    "nan": Severity.ERROR,
    "inf": Severity.ERROR,
    "high_low": Severity.ERROR,
    "high": Severity.ERROR,
    "low": Severity.ERROR,
    "timestamp": Severity.ERROR,
    "gaps": Severity.WARNING,
    "price": Severity.WARNING,
    "extreme_move": Severity.WARNING,
    "duplicate_dates": Severity.ERROR,
    "non_trading_day": Severity.ERROR,
    "split_jump": Severity.ERROR,
    "source": Severity.ERROR,
}


@dataclass
class ValidationConfig:
    """Per-check configuration for :func:`validate_ohlc`.

    Any entry in ``severity`` overrides :data:`DEFAULT_SEVERITIES` for that
    check, so a caller can downgrade a corruption class to a warning (or
    upgrade an existing warning to an error).
    """
    severity: Dict[str, Severity] = field(default_factory=dict)
    #: Single-bar |return| above this looks like an unadjusted split.
    split_jump_threshold: float = 0.5
    #: Enable weekend/holiday bar detection (off by default: some instruments
    #: trade weekends). ``trading_days`` (weekday numbers, Mon=0) or ``holidays``
    #: also enable it.
    check_trading_calendar: bool = False
    trading_days: Optional[Set[int]] = None
    holidays: Optional[Set] = None
    #: Provenance column for source-detection; checked when present (or required).
    provenance_column: str = "source"
    require_provenance: bool = False
    #: ``None`` = auto (only for daily-like series), True/False = force.
    check_duplicate_dates: Optional[bool] = None

    def __post_init__(self) -> None:
        merged = dict(DEFAULT_SEVERITIES)
        merged.update(self.severity)
        self.severity = merged

    def sev(self, name: str) -> Severity:
        return self.severity.get(name, Severity.ERROR)


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


def _add(
    report: ValidationReport,
    config: ValidationConfig,
    name: str,
    field: str,
    message: str,
    row_indices: Optional[List[int]] = None,
) -> None:
    """Append an issue at the configured severity and bump the right counter."""
    severity = config.sev(name)
    report.issues.append(ValidationIssue(severity, field, message, row_indices))
    if severity is Severity.ERROR:
        report.n_errors += 1
    else:
        report.n_warnings += 1


def _is_daily_like(ts_index: pd.DatetimeIndex, expected_freq: Optional[str]) -> bool:
    """True if the series is daily or coarser (duplicate calendar dates matter)."""
    if expected_freq is not None:
        try:
            return pd.Timedelta(expected_freq) >= pd.Timedelta(hours=20)
        except (ValueError, TypeError):
            return False
    if len(ts_index) > 2:
        return (ts_index[1:] - ts_index[:-1]).median() >= pd.Timedelta(hours=20)
    return False


def validate_ohlc(
    df: pd.DataFrame,
    expected_freq: Optional[str] = None,
    max_gap_multiple: int = 3,
    config: Optional[ValidationConfig] = None,
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
    config : ValidationConfig, optional
        Per-check severity overrides and thresholds. Defaults to ERROR for the
        corruption classes (duplicate dates, non-trading-day bars, split-like
        jumps, mixed/missing source).

    Returns
    -------
    ValidationReport
    """
    config = config or ValidationConfig()
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
        _add(report, config, "columns", "columns", f"Missing required columns: {missing}")
        report.n_bars = len(df)
        return report  # Can't proceed without OHLC columns

    report.n_bars = len(df)
    if report.n_bars == 0:
        _add(report, config, "data", "data", "DataFrame is empty")
        return report

    ohlc = df[["open", "high", "low", "close"]]

    # 2. Numeric types (use api.types to handle pandas extension dtypes like StringDtype)
    non_numeric = [col for col in ohlc.columns if not pd.api.types.is_numeric_dtype(ohlc[col])]
    if non_numeric:
        _add(report, config, "dtype", "dtype", f"Non-numeric OHLC columns: {non_numeric}")
        return report

    # 3. NaN / inf
    nan_mask = ohlc.isna().any(axis=1)
    if nan_mask.any():
        indices = list(np.where(nan_mask)[0][:20])  # Cap at 20 for readability
        _add(report, config, "nan", "nan", "NaN values in OHLC data", indices)

    inf_mask = np.isinf(ohlc.to_numpy()).any(axis=1)
    if inf_mask.any():
        indices = list(np.where(inf_mask)[0][:20])
        _add(report, config, "inf", "inf", "Infinite values in OHLC data", indices)

    # 4. OHLC consistency
    h, l, o_col, c_col = ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), ohlc["open"].to_numpy(), ohlc["close"].to_numpy()

    hl_bad = np.where(h < l)[0]
    if len(hl_bad) > 0:
        _add(report, config, "high_low", "high_low", f"high < low on {len(hl_bad)} bars",
             list(hl_bad[:20]))

    ho_bad = np.where(h < o_col)[0]
    hc_bad = np.where(h < c_col)[0]
    high_bad = np.union1d(ho_bad, hc_bad)
    if len(high_bad) > 0:
        _add(report, config, "high", "high",
             f"high < open or high < close on {len(high_bad)} bars", list(high_bad[:20]))

    lo_bad = np.where(l > o_col)[0]
    lc_bad = np.where(l > c_col)[0]
    low_bad = np.union1d(lo_bad, lc_bad)
    if len(low_bad) > 0:
        _add(report, config, "low", "low",
             f"low > open or low > close on {len(low_bad)} bars", list(low_bad[:20]))

    # 5-8: Timestamp / calendar checks (only if we have timestamps)
    if ts_index is not None and len(ts_index) > 1:
        # 5. Duplicate timestamps
        dup_mask = ts_index.duplicated()
        if dup_mask.any():
            indices = list(np.where(dup_mask)[0][:20])
            _add(report, config, "timestamp", "timestamp",
                 f"{dup_mask.sum()} duplicate timestamps", indices)

        # 6. Duplicate calendar dates (daily-frequency only unless forced):
        #    a date appearing more than once means a doubled/overwritten series.
        check_dup_dates = config.check_duplicate_dates
        if check_dup_dates is True or (
            check_dup_dates is None and _is_daily_like(ts_index, expected_freq)
        ):
            date_mask = pd.Index(ts_index.date).duplicated()
            if date_mask.any():
                indices = list(np.where(date_mask)[0][:20])
                _add(report, config, "duplicate_dates", "duplicate_dates",
                     f"{int(date_mask.sum())} duplicate calendar dates", indices)

        # 7. Monotonicity
        if not ts_index.is_monotonic_increasing:
            diffs = ts_index[1:] - ts_index[:-1]
            non_mono = np.where(diffs < pd.Timedelta(0))[0]
            _add(report, config, "timestamp", "timestamp",
                 f"Non-monotonic timestamps at {len(non_mono)} points",
                 list(non_mono[:20]))

        # 8. Non-trading-day bars (opt-in: some instruments trade weekends).
        if config.check_trading_calendar or config.trading_days is not None or config.holidays:
            allowed = config.trading_days if config.trading_days is not None else {0, 1, 2, 3, 4}
            bad = ~np.isin(ts_index.dayofweek.to_numpy(), list(allowed))
            if config.holidays:
                holiday_set = {pd.Timestamp(d).date() for d in config.holidays}
                bad = bad | pd.Index(ts_index.date).isin(holiday_set)
            if bad.any():
                indices = list(np.where(bad)[0][:20])
                _add(report, config, "non_trading_day", "non_trading_day",
                     f"{int(bad.sum())} bars on non-trading days", indices)

        # 9. Gap detection
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
                    _add(report, config, "gaps", "gaps",
                         f"{gap_mask.sum()} gaps exceeding {max_gap_multiple}x expected frequency ({freq_td})",
                         gap_indices)

    # 10. Zero/negative prices
    nonpos = (ohlc <= 0).any(axis=1)
    if nonpos.any():
        indices = list(np.where(nonpos)[0][:20])
        _add(report, config, "price", "price",
             f"{nonpos.sum()} bars with zero or negative prices", indices)

    # 11. Extreme moves (kept as a warning; see split_jump below for the
    #     corruption-grade check)
    if len(c_col) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.abs(np.diff(c_col) / c_col[:-1])
        extreme = np.where(returns > 0.5)[0]
        if len(extreme) > 0:
            _add(report, config, "extreme_move", "extreme_move",
                 f"{len(extreme)} bars with >50% single-bar move (possible bad data)",
                 list(extreme[:20]))

        # 12. Split-like jumps: an implausibly large single-bar move is the
        #     signature of an unadjusted split. ERROR by default, configurable.
        jumps = np.where(returns > config.split_jump_threshold)[0]
        if len(jumps) > 0:
            _add(report, config, "split_jump", "split_jump",
                 f"{len(jumps)} bars with >{config.split_jump_threshold:.0%} single-bar "
                 f"move (possible unadjusted split)", list(jumps[:20]))

    # 13. Mixed / missing provenance. Only meaningful for sources that carry a
    #     provenance column; a mixed or partial column means the frame is a
    #     blend of sources and must not be trusted.
    source_col = config.provenance_column
    if source_col in df.columns:
        series = df[source_col]
        non_null = series.dropna()
        problems = []
        if non_null.nunique() > 1:
            problems.append(f"mixed source values {sorted(map(str, non_null.unique()))[:5]}")
        missing_count = int(series.isna().sum())
        if missing_count:
            problems.append(f"{missing_count} rows missing {source_col}")
        if problems:
            _add(report, config, "source", source_col, "; ".join(problems))
    elif config.require_provenance:
        _add(report, config, "source", source_col,
             f"required provenance column {source_col!r} is missing")

    return report


def validate_ticks(
    ticks,
) -> ValidationReport:
    """Validate a list of Tick objects for common data quality issues.

    Parameters
    ----------
    ticks : list of Tick
        Raw tick data to validate.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    report.n_bars = len(ticks)  # Reuse n_bars field for tick count

    if len(ticks) == 0:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "data", "Tick list is empty"))
        report.n_errors += 1
        return report

    # 1. NaN / inf prices
    bad_prices = [i for i, t in enumerate(ticks)
                  if t.price != t.price or np.isinf(t.price)]  # NaN != NaN
    if bad_prices:
        report.issues.append(ValidationIssue(
            Severity.ERROR, "price", f"NaN/inf prices at {len(bad_prices)} ticks",
            row_indices=bad_prices[:20]))
        report.n_errors += 1

    # 2. Non-positive prices
    nonpos = [i for i, t in enumerate(ticks) if t.price <= 0]
    if nonpos:
        report.issues.append(ValidationIssue(
            Severity.WARNING, "price", f"{len(nonpos)} ticks with zero or negative price",
            row_indices=nonpos[:20]))
        report.n_warnings += 1

    # 3. Monotonic timestamps
    if len(ticks) > 1:
        non_mono = []
        for i in range(1, len(ticks)):
            if ticks[i].ts < ticks[i - 1].ts:
                non_mono.append(i)
                if len(non_mono) >= 20:
                    break
        if non_mono:
            report.issues.append(ValidationIssue(
                Severity.ERROR, "timestamp",
                f"Non-monotonic timestamps at {len(non_mono)}+ points",
                row_indices=non_mono))
            report.n_errors += 1

    # 4. Duplicate timestamps (warning, not error — multiple ticks at same
    # millisecond is normal for high-frequency data)
    if len(ticks) > 1:
        dup_count = sum(1 for i in range(1, len(ticks)) if ticks[i].ts == ticks[i - 1].ts)
        if dup_count > 0:
            report.issues.append(ValidationIssue(
                Severity.WARNING, "timestamp",
                f"{dup_count} consecutive duplicate timestamps"))
            report.n_warnings += 1

    return report
