from __future__ import annotations

# ruff: noqa: PD901
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hpo_glue.plotting.styles import categorical_colors, distinct_markers

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

"""
# What else would I need in here?

Data columns (total 25 columns):
 #   Column                        Non-Null Count  Dtype
---  ------                        --------------  -----
 0   config.id                     360 non-null    string
 1   query.id                      360 non-null    string
 2   result.budget_cost            360 non-null    Float32
 3   query.fidelity.count          360 non-null    UInt8
 4   query.fidelity.1.name         360 non-null    category
 5   query.fidelity.1.value        360 non-null    UInt8
 6   result.objective.1.value      360 non-null    Float32
 7   result.fidelity.1.value       360 non-null    UInt8
 8   problem.name                  360 non-null    category
 9   problem.objective.count       360 non-null    UInt8
 10  problem.objective.1.name      360 non-null    category
 11  problem.objective.1.minimize  360 non-null    boolean
 12  problem.objective.1.min       360 non-null    Float32
 13  problem.objective.1.max       360 non-null    Float32
 14  problem.fidelity.count        360 non-null    UInt8
 15  problem.fidelity.1.name       360 non-null    category
 16  problem.fidelity.1.min        360 non-null    UInt8
 17  problem.fideltiy.1.max        360 non-null    UInt8
 18  problem.cost.count            360 non-null    UInt8
 19  problem.benchmark             360 non-null    category
 20  problem.seed                  360 non-null    UInt8
 21  problem.budget.kind           360 non-null    category
 22  problem.budget.total          360 non-null    UInt8
 23  problem.opt.name              360 non-null    category
 24  problem.opt.hp_str            360 non-null    category
dtypes: Float32(4), UInt8(10), boolean(1), category(8), string(2)
memory usage: 24.2 KB
"""


def _incumbents(
    df: pd.DataFrame,
    *,
    by: str,
    minimize: bool,
) -> pd.DataFrame:
    return (
        df.assign(cumulative=(df[by].cummin() if minimize else df[by].cummax()))
        .drop_duplicates(subset="cumulative", keep="first")
        .drop(columns="cumulative")  # type: ignore
    )


def _join_if_tuple(
    x: Any | tuple[Any, ...],
    *,
    sep: str,
) -> str:
    if not isinstance(x, tuple):
        return str(x)
    return sep.join(map(str, x))


def _normalize_objectives_as_loss(
    df: pd.DataFrame,
    *,
    objectives: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    for name, (min_val, max_val) in objectives.items():
        df[name] = (df[name] - min_val) / (max_val - min_val)
    return df


def _plot_one_benchmark(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    minimize: bool,
    hue: str | Sequence[str],
    marker: str | Sequence[str],
    seed: str | None,
    ylim: tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    xlabel: str | None,
    ylabel: str | None,
    ax: plt.Axes,
) -> None:
    _regret_bound = df[y].min() if minimize else df[y].max()
    colors_needed = len(df[hue].drop_duplicates())
    colors = categorical_colors(n=colors_needed)

    markers_needed = len(df[marker].drop_duplicates())
    markers = distinct_markers(n=markers_needed)

    for (_hue_keys, hue_df), _hue in zip(df.groupby(hue), colors, strict=True):
        for (_marker_keys, marker_df), _marker in zip(hue_df.groupby(marker), markers, strict=True):
            hue_label = _join_if_tuple(_hue_keys, sep=", ")
            marker_label = _join_if_tuple(_marker_keys, sep=", ")
            label = f"{hue_label} ({marker_label})"

            selected_df = marker_df
            selected_df = selected_df.sort_values(by=x)

            full_frame = pd.DataFrame()
            for seed_key, seed_df in selected_df.groupby(seed):
                only_incs = _incumbents(seed_df, by=y, minimize=minimize)
                inc_trace = only_incs.set_index(x)[y]
                inc_trace = inc_trace - _regret_bound if minimize else _regret_bound - inc_trace
                inc_trace.name = f"seed-{seed_key}"
                full_frame[seed_key] = inc_trace

            _data = full_frame.sort_index().ffill().dropna()
            _data = _data.agg(["mean", "std"], axis=1)

            _xs = _data.index.astype(np.float32).to_numpy()
            _means = _data["mean"].to_numpy()  # type: ignore
            _stds = _data["std"].to_numpy()  # type: ignore

            ax.plot(  # type: ignore
                _xs,
                _means,
                drawstyle="steps-post",
                label=label,
                linestyle="solid",  # type: ignore
                markevery=10,
                marker=_marker,
                linewidth=3,
            )

            ax.fill_between(
                _xs,
                _means - _stds,
                _means + _stds,
                alpha=0.2,
                color=_hue,
                edgecolor=_hue,
                linewidth=2,
                step="post",
            )

    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
