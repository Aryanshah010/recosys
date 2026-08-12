from __future__ import annotations

from typing import Any, cast

import pandas as pd


def mean_frame(
    df: pd.DataFrame,
    by: str | list[str],
    columns: list[str],
    decimals: int = 4,
) -> pd.DataFrame:
    grouped = df.groupby(by)[columns].mean()
    return cast(pd.DataFrame, grouped).round(decimals)


def mean_series(
    df: pd.DataFrame,
    by: str | list[str],
    column: str,
    decimals: int = 4,
) -> pd.Series:
    grouped = df.groupby(by)[column].mean()
    return cast(pd.Series, grouped).round(decimals)


def agg_frame(
    df: pd.DataFrame,
    by: str | list[str],
    columns: list[str],
    funcs: list[str],
    decimals: int = 4,
) -> pd.DataFrame:
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
    table = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    return cast(pd.DataFrame, table).round(decimals)


def as_frame(obj: Any) -> pd.DataFrame:
    return cast(pd.DataFrame, obj)


def as_series(obj: Any) -> pd.Series:
    return cast(pd.Series, obj)


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value is pd.NA:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value is pd.NA:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
