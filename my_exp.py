from __future__ import annotations

import logging

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.experiment import Experiment
from hpo_glue.optimizers.dehb import DEHB_Optimizer
from hpo_glue.optimizers.smac import SMAC_Hyperband

# logging.basicConfig(level=logging.INFO)
non_tabular_benchmarks = [b for b in BENCHMARKS.values() if not b.is_tabular]

logger = logging.getLogger("smac")
logger.setLevel(logging.ERROR)

# Generate single objective problems across all non-tabular benchmarks
# for SMAC_BO and Optuna with different hyperparameters
experiment = Experiment.generate(
    optimizers=[DEHB_Optimizer, SMAC_Hyperband],
    benchmarks=[BENCHMARKS["mfh3_good"], BENCHMARKS["mfh6_good"]],
    budget=50,
    objectives=1,
    fidelities=1,
    seeds=range(2),
    on_error="ignore",
)
experiment_report = experiment.run(overwrite=True)
# df = experiment_report.df()
# print(df)
# print(df.info())
"""
from hpo_glue.plotting.incumbent_trace import _plot_one_benchmark

fig, ax = plt.subplots()
_plot_one_benchmark(
    df,
    x="result.budget_used_total",
    y="result.objective.1.value",
    minimize=True,
    hue="problem.opt.name",
    marker="problem.opt.hp_str",
    seed="problem.seed",
    ax=ax,
    xlim=(0, 50),
    ylim=None,
    xlabel="Budget Used",
    ylabel="Objective Value",
)
plt.show()
"""
