from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataContainer: TypeAlias = np.ndarray | (pd.DataFrame | pd.Series)
D = TypeVar("D", bound=DataContainer)


def plot_results(  # noqa: PLR0915
    *,
    report: dict[str, Any],
    budget_type: str,
    budget: int,
    objective: str,
    minimize: bool,
    save_dir: Path,
    benchmarks_name: str,
) -> None:
    """Plot the results for the optimizers on the given benchmark."""
    marker_list = [
        "o",
        "X",
        "^",
        "H",
        ">",
        "^",
        "p",
        "P",
        "*",
        "h",
        "<",
        "s",
        "x",
        "+",
        "D",
        "d",
        "|",
        "_",
    ]
    markers = cycle(marker_list)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
    colors_mean = cycle(colors)
    optimizers = list(report.keys())
    plt.figure(figsize=(20, 10))
    optim_res_dict = {}
    for instance in optimizers:
        optim_res_dict[instance] = {}
        seed_cost_dict = {}
        budgets = []
        for seed in report[instance]:
            results = report[instance][seed]["results"]
            cost_list = results[results["objectives"][0]].values.astype(np.float64)
            if budget_type == "fidelity_budget":
                budget_list = report[instance][seed]["results"]["fidelity"].values.astype(
                    np.float64
                )
                if np.isnan(budget_list[0]):
                    budget_list = np.cumsum(
                        np.repeat(float(results["max_budget"][0]), len(budget_list))
                    )
                    if len(budget_list) > len(budgets):
                        budgets = budget_list
                else:
                    budget_list = np.cumsum(budget_list)
            elif budget_type == "n_trials":
                budget_list = np.arange(1, budget + 1)
            else:
                raise NotImplementedError(f"Budget type {budget_type} not implemented")

            seed_cost_dict[seed] = pd.Series(cost_list, index=budget_list)
        seed_cost_df = pd.DataFrame(seed_cost_dict)
        seed_cost_df = seed_cost_df.ffill(axis=0)
        seed_cost_df = seed_cost_df.dropna(axis=0)
        means = pd.Series(seed_cost_df.mean(axis=1), name=f"means_{instance}")
        std = pd.Series(seed_cost_df.std(axis=1), name=f"std_{instance}")
        optim_res_dict[instance]["means"] = means
        optim_res_dict[instance]["std"] = std
        means = means.cummin() if minimize else means.cummax()
        means = means.drop_duplicates()
        std = std.loc[means.index]
        means[budget] = means.iloc[-1]
        std[budget] = std.iloc[-1]
        col_next = next(colors_mean)
        plt.step(
            means.index,
            means,
            where="post",
            label=instance,
            marker=next(markers),
            markersize=10,
            markerfacecolor="#ffffff",
            markeredgecolor=col_next,
            markeredgewidth=2,
            color=col_next,
            linewidth=3,
        )
        plt.fill_between(
            means.index,
            means - std,
            means + std,
            alpha=0.2,
            step="post",
            color=col_next,
            edgecolor=col_next,
            linewidth=2,
        )
    plt.xlabel(f"{budget_type}")
    plt.ylabel(f"{objective}")
    plt.title(f"Performance of Optimizers on {benchmarks_name}")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {benchmarks_name}")
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{benchmarks_name}_performance.png")
    plt.show()
    optim_means = pd.DataFrame({k: v["means"] for k, v in optim_res_dict.items()})
    optim_stds = pd.DataFrame({k: v["std"] for k, v in optim_res_dict.items()})
    if len(optimizers) > 1:
        plot_by_ranking(
            optim_means=optim_means,
            optim_stds=optim_stds,
            budget_type=budget_type,
            benchmarks_name=benchmarks_name,
        )


def plot_by_ranking(
    *,
    optim_means: pd.DataFrame,
    optim_stds: pd.DataFrame,
    budget_type: str,
    benchmarks_name: str,
) -> None:
    """Plots the results by ranking the optimizers."""
    optim_means = optim_means.ffill(axis=0)
    optim_means = optim_means.dropna(axis=0)
    optim_stds = optim_stds.ffill(axis=0)
    optim_stds = optim_stds.dropna(axis=0)

    rankings = optim_means.rank(ascending=False, axis=1, method="min")

    # Plot the rankings
    plt.figure(figsize=(20, 10))
    for optimizer in optim_means.columns:
        plt.plot(rankings.index, rankings[optimizer], label=optimizer)

    # Customize the plot
    plt.title(f"Optimizer Rankings Over {budget_type} on {benchmarks_name}")
    plt.xlabel(budget_type)
    plt.ylabel("Rankings")
    plt.legend()
    plt.grid(visible=True)
    plt.show()


def agg_data(exp_dir: str | Path) -> None:
    """Aggregate the data from the run directory for plotting."""
    exp_dir = Path(exp_dir)
    for runs in exp_dir.iterdir():
        for bench_dir in runs.iterdir():
            if bench_dir == "plots":
                continue

            df_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            budget_type: str | None = None
            budget: int | None = None
            objective: str | None = None
            minimize = True
            for opt_dir in bench_dir.iterdir():
                for seed in opt_dir.iterdir():
                    for file in seed.iterdir():
                        res_df = pd.read_parquet(file)

                        objective = res_df[["objectives"]].iloc[0].values[0]
                        budget_type = res_df[["budget_type"]].iloc[0].values[0]
                        budget = int(res_df[["budget"]].iloc[0].values[0])
                        minimize = res_df[["minimize"]].iloc[0].values[0]

                        res_df = res_df[[objective, "fidelity", "max_budget", "objectives"]]
                        instance = file.stem.split(bench_dir.name)[-1][1:]
                        instance = instance[:-1] if instance[-1] == "_" else instance
                        df_agg[instance][int(seed.name)] = {"results": res_df}  # type: ignore

            assert budget_type is not None
            assert budget is not None
            assert objective is not None
            plot_results(
                report=df_agg,
                budget_type=budget_type,
                budget=budget,
                objective=objective,
                minimize=minimize,
                save_dir=runs / "plots",
                benchmarks_name=bench_dir.name,
            )


def reduce_floating_precision(x: D) -> D:
    """Reduce the floating point precision of the data.

    For a float array, will reduce by one step, i.e. float32 -> float16, float64
    -> float32.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_floating_precision(x[col])
        return x  # type: ignore

    if x.dtype.kind != "f":
        return x

    _reduction_map = {
        # Base numpy dtypes
        "float128": "float64",
        "float96": "float64",
        "float64": "float32",
        "float32": "float16",
        # Nullable pandas dtypes (only supports 64 and 32 bit)
        "Float64": "Float32",
    }

    if (dtype := _reduction_map.get(x.dtype.name)) is not None:
        return x.astype(dtype)  # type: ignore

    return x


def reduce_int_span(x: D) -> D:
    """Reduce the integer span of the data.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_int_span(x[col])
        return x  # type: ignore

    if x.dtype.kind not in "iu":
        return x

    min_dtype = np.min_scalar_type(x.min())  # type: ignore
    max_dtype = np.min_scalar_type(x.max())  # type: ignore
    dtype = np.result_type(min_dtype, max_dtype)

    # The above dtype is a numpy dtype and may not allow for nullable values,
    # which are permissible in pandas. `to_numeric` will convert to appropriate
    # pandas nullable dtypes.
    if isinstance(x, pd.Series):
        dc = "unsigned" if "uint" in dtype.name else "integer"
        return pd.to_numeric(x, downcast=dc)

    return x.astype(dtype)


def reduce_dtypes(x: D, *, reduce_int: bool = True, reduce_float: bool = True) -> D:
    """Reduce the dtypes of data.

    When a dataframe, will reduce the dtypes of all columns.
    When applied to an iterable, will apply to all elements of the iterable.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array. Otherwise for floats, will reduce
    by one step, i.e. float32 -> float16, float64 -> float32.

    Args:
        x: The data to reduce.
        reduce_int: Whether to reduce integer dtypes.
        reduce_float: Whether to reduce floating point dtypes.
    """
    if not isinstance(x, pd.DataFrame | pd.Series | np.ndarray):
        raise TypeError(f"Cannot reduce data of type {type(x)}.")

    if isinstance(x, pd.Series | pd.DataFrame):
        x = x.convert_dtypes()

    if reduce_int:
        x = reduce_int_span(x)
    if reduce_float:
        x = reduce_floating_precision(x)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Incumbents after GLUE Experiments")

    parser.add_argument(
        "--root_dir", type=Path, help="Location of the root directory", default=Path("./")
    )

    parser.add_argument(
        "--results_dir", type=str, help="Location of the results directory", default="./results"
    )

    parser.add_argument(
        "--exp_dir", type=str, help="Location of the Experiment directory", default=None
    )

    parser.add_argument("--save_dir", type=str, help="Directory to save the plots", default="plots")

    args = parser.parse_args()

    if args.exp_dir is None:
        raise ValueError("Experiment directory not specified")

    exp_dir = args.root_dir / args.results_dir / args.exp_dir

    agg_data(exp_dir)
