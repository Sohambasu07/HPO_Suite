from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import os
import pandas as pd
import numpy as np
import pprint
from itertools import cycle
import random
from collections import defaultdict

from hpo_glue.glu import GLUEReport

def plot_results(
        report: Dict[str, Any],
        budget_type: str,
        budget: int,
        objective: str,
        minimize: bool,
        save_dir: Path,
        benchmarks_name: str,
):
    """Plot the results for the optimizers on the given benchmark"""
    
    marker_list = ["o", "X", "^", "H", ">", "^", "p", "P", "*", "h", "<", "s", "x", "+", "D", "d", "|", "_"]
    # random.shuffle(marker_list)
    markers = cycle(marker_list)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_mean = cycle(colors)
    optimizers = list(report.keys())
    print(f"Plotting performance of Optimizers on {benchmarks_name}")
    plt.figure(figsize=(20, 10))
    optim_res_dict = dict()
    for instance in optimizers:
        print(instance)
        optim_res_dict[instance] = dict()
        seed_cost_dict = dict()
        budgets = []
        for seed in report[instance].keys():
            results = report[instance][seed]["results"]
            cost_list = results[results["objectives"][0]].values.astype(np.float64)
            budget_list = report[instance][seed]["results"]["fidelity"].values.astype(np.float64)
            if np.isnan(budget_list[0]):
                budget_list = np.cumsum(np.repeat(float(results["max_budget"][0]), len(budget_list)))
                if len(budget_list) > len(budgets):
                    budgets = budget_list
            else:
                budget_list = np.cumsum(budget_list)
            seed_cost_dict[seed] = pd.Series(cost_list, index = budget_list)
        seed_cost_df = pd.DataFrame(seed_cost_dict)
        seed_cost_df.ffill(axis = 0, inplace = True)
        seed_cost_df.dropna(axis = 0, inplace = True)
        means = pd.Series(seed_cost_df.mean(axis=1), name = f"means_{instance}")
        std = pd.Series(seed_cost_df.std(axis=1), name = f"std_{instance}")
        optim_res_dict[instance]["means"] = means
        optim_res_dict[instance]["std"] = std
        if minimize:
            means = means.cummin()
        else:
            means = means.cummax()
        means = means.drop_duplicates()
        std = std.loc[means.index]
        means[budget] = means.iloc[-1]
        std[budget] = std.iloc[-1]
        col_next = next(colors_mean)
        plt.step(
            means.index, 
            means, 
            where = 'post', 
            label = instance,
            marker = next(markers),
            markersize = 10,
            markerfacecolor = '#ffffff',
            markeredgecolor = col_next,
            markeredgewidth = 2,
            color = col_next,
            linewidth = 3
        )
        plt.fill_between(
            means.index, 
            means - std, 
            means + std, 
            alpha = 0.2,
            step = 'post',
            color = col_next,
            edgecolor = col_next,
            linewidth = 2            
        )
    plt.xlabel(f"{budget_type}")
    plt.ylabel(f"{objective}")
    plt.title(f"Performance of Optimizers on {benchmarks_name}")
    plt.legend()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir / f"{benchmarks_name}_performance.png")
    plt.show()
    optim_means = pd.DataFrame({k: v["means"] for k, v in optim_res_dict.items()})
    optim_stds = pd.DataFrame({k: v["std"] for k, v in optim_res_dict.items()})
    plot_by_ranking(
        optim_means = optim_means,
        optim_stds = optim_stds,
        budget_type = budget_type,
        budget = budget,
        objective = objective,
        minimize = minimize,
        save_dir = save_dir,
        benchmarks_name = benchmarks_name,
    )


def plot_by_ranking(
        optim_means: pd.DataFrame,
        optim_stds: pd.DataFrame,
        budget_type: str,
        budget: int,
        objective: str,
        minimize: bool,
        save_dir: Path,
        benchmarks_name: str,
):
    """Plots the results by ranking the optimizers"""
    print("Plotting by Ranking")
    # print(optim_means)
    # print(optim_stds)
    optim_means.ffill(axis = 0, inplace = True)
    optim_means.dropna(axis = 0, inplace = True)
    optim_stds.ffill(axis = 0, inplace = True)
    optim_stds.dropna(axis = 0, inplace = True)

    rankings = optim_means.rank(ascending=False, axis=1, method='min')

    # Plot the rankings
    plt.figure(figsize=(20, 10))
    for optimizer in optim_means.columns:
        plt.plot(rankings.index, rankings[optimizer], label=optimizer)

    # Customize the plot
    plt.title(f'Optimizer Rankings Over {budget_type} on {benchmarks_name}')
    plt.xlabel(budget_type)
    plt.ylabel('Rankings')
    plt.legend()
    plt.grid(True)
    plt.show()


def agg_data(
        exp_dir: str,
):
    """Aggregate the data from the run directory for plotting"""

    for runs in os.listdir(exp_dir):
        for bench_dir in os.listdir(exp_dir/ runs):
            if bench_dir == "plots":
                continue
            # if "tabular" in bench_dir:
            #     continue
            df_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            opts = os.listdir(exp_dir / runs / bench_dir)
            budget_type = None
            budget = None
            objective = None
            minimize = True
            for i, opt_dir in enumerate(opts):
                for seed in os.listdir(exp_dir/ runs / bench_dir / opt_dir):
                    files = os.listdir(exp_dir/ runs / bench_dir / opt_dir / seed)
                    for file in files:
                        res_df = pd.read_parquet(exp_dir/ runs / bench_dir / opt_dir / seed /file)
                        objective = res_df[['objectives']].iloc[0].values[0]
                        budget_type = res_df[['budget_type']].iloc[0].values[0]
                        budget = res_df[['budget']].iloc[0].values[0]
                        minimize = res_df[['minimize']].iloc[0].values[0]
                        res_df = res_df[[objective, 'fidelity', 'max_budget', 'objectives']]
                        instance = (file.split(".parquet")[0]).split(bench_dir)[-1][1:]
                        instance = instance[:-1] if instance[-1] == "_" else instance
                        df_agg[instance][int(seed)] = {
                            "results": res_df
                        }
            # pprint.pprint(df_agg, indent=4)
            # exit(0)
            # print("Keys: ", df_agg.keys())
            plot_results(
                report = df_agg,
                budget_type = budget_type,
                budget = budget,
                objective = objective,
                minimize = minimize,
                save_dir = exp_dir / runs / "plots",
                benchmarks_name = bench_dir,
            )
                               


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Incumbents after GLUE Experiments")

    parser.add_argument("--root_dir",
                        type=Path,
                        help="Location of the root directory",
                        default=Path("./"))

    parser.add_argument("--results_dir",
                        type=str,
                        help="Location of the results directory",
                        default="./results")
    
    parser.add_argument("--exp_dir",
                        type=str,
                        help="Location of the Experiment directory",
                        default=None)
    
    parser.add_argument("--save_dir",
                        type=str,
                        help="Directory to save the plots",
                        default="plots")

    args = parser.parse_args()

    if args.exp_dir is None:
        raise ValueError("Experiment directory not specified")
    
    exp_dir = args.root_dir / args.results_dir / args.exp_dir
    
    agg_data(exp_dir)


