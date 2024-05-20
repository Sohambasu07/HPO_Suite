from __future__ import annotations

# ruff: noqa: PD901
from collections.abc import Sequence
from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd

from hpo_glue.plotting.styles import MARKERS

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


def _plot_one_benchmark(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_requires_cumulation: bool,
    minimize: bool,
    hue: str | Sequence[str],
    marker: str | Sequence[str],
    seed: str | None,
    ax: plt.Axes,
) -> None:
    if x_requires_cumulation:
        df = df.sort_values(x)
        assert "x.cumulated" not in df.columns
        df["x.cumulated"] = df[x].cumsum()
        x = "x.cumulated"

    if minimize:
        regret_bound = df[y].min()
        df["y.regret"] = df[y] - regret_bound
    else:
        regret_bound = df[y].max()
        df["y.regret"] = regret_bound - df[y]

    colors_needed = len(df[hue].drop_duplicates())
    if colors_needed <= 10:  # noqa: PLR2004
        _hues = iter(plt.get_cmap("tab10").colors)  # type: ignore
    else:
        _hues = cycle(plt.get_cmap("tab20").colors)  # type: ignore

    for (_hue_keys, hue_df), _hue in zip(df.groupby(hue), _hues, strict=False):
        for (_marker_keys, marker_df), _marker in zip(
            hue_df.groupby(marker), cycle(MARKERS), strict=False
        ):
            (", ".join(map(str, _hue_keys)) if isinstance(_hue_keys, tuple) else str(_hue_keys))
            (
                ", ".join(map(str, _marker_keys))
                if isinstance(_marker_keys, tuple)
                else str(_marker_keys)
            )

            full_frame = pd.DataFrame()
            for seed_key, seed_df in marker_df.groupby(seed):
                full_frame[seed_key] = (
                    seed_df.assign(cumulative=seed_df["y.regret"].cummin())
                    .drop_duplicates(subset="cumulative", keep="first")
                    .drop(columns="cumulative")  # type: ignore
                    .set_index(x)["y.regret"]
                )

            _data = full_frame.sort_index().ffill().dropna().agg(["mean", "std"], axis=1)
            _means = _data["mean"]
            _stds = _data["std"]

            _means.plot(  # type: ignore
                drawstyle="steps-post",
                label=f"{y} (regret)",
                ax=ax,
                linestyle="solid",  # type: ignore
                markevery=10,
                marker=_marker,
                linewidth=3,
            )
            ax.fill_between(
                _data.index,  # type: ignore
                _means - _stds,
                _means + _stds,
                alpha=0.2,
                color=_hue,
                edgecolor=_hue,
                marker=_marker,
                linewidth=2,
                step="post",
            )
