"""Typed pandas aggregation helpers used by the evaluation and reporting code.

Grouped-aggregate-then-round is the single most repeated operation in this
project. Centralising it removes the duplication and gives the result a
concrete static type: pandas' own stubs widen `groupby(...)[col].mean()` to a
`Scalar` union, so chained calls such as `.round()` cannot be resolved by a
type checker even though they are valid at runtime.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def mean_frame(
    df: pd.DataFrame,
    by: str | list[str],
    columns: list[str],
    decimals: int = 4,
) -> pd.DataFrame:
    """Grouped mean over several columns, rounded, as a DataFrame."""
    grouped = df.groupby(by)[columns].mean()
    return cast(pd.DataFrame, grouped).round(decimals)


def mean_series(
    df: pd.DataFrame,
    by: str | list[str],
    column: str,
    decimals: int = 4,
) -> pd.Series:
    """Grouped mean over one column, rounded, as a Series."""
    grouped = df.groupby(by)[column].mean()
    return cast(pd.Series, grouped).round(decimals)


def agg_frame(
    df: pd.DataFrame,
    by: str | list[str],
    columns: list[str],
    funcs: list[str],
    decimals: int = 4,
) -> pd.DataFrame:
    """Grouped multi-statistic aggregate with flattened `metric_stat` columns."""
    aggregated = cast(pd.DataFrame, df.groupby(by)[columns].agg(funcs)).round(decimals)
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    return aggregated


def pivot_mean(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    decimals: int = 3,
) -> pd.DataFrame:
    """Mean of `values` pivoted into an index x columns table."""
    table = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    return cast(pd.DataFrame, table).round(decimals)


def as_frame(obj: Any) -> pd.DataFrame:
    """Assert a pandas result is a DataFrame for static analysis."""
    return cast(pd.DataFrame, obj)


def as_series(obj: Any) -> pd.Series:
    """Assert a pandas result is a Series for static analysis."""
    return cast(pd.Series, obj)


def to_int(value: Any, default: int = 0) -> int:
    """Coerce a pandas scalar or index label to int, tolerating NA."""
    if value is None or value is pd.NA:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a pandas scalar to float, tolerating NA."""
    if value is None or value is pd.NA:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
