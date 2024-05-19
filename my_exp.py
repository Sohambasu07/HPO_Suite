from __future__ import annotations

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.optimizers.smac import SMAC_Hyperband
from hpo_glue.problem import Problem

non_tabular_benchmarks = [b for b in BENCHMARKS.values() if not b.is_tabular]

# Generate single objective problems across all non-tabular benchmarks
# for SMAC_BO and Optuna with different hyperparameters
problems = Problem.generate(
    optimizers=SMAC_Hyperband,
    benchmarks=BENCHMARKS["mfh3_good"],
    budget=25,
    objectives=1,
    fidelities=1,
    seeds=range(1),
    on_error="ignore",
)
for problem in problems:
    report = problem.run(overwrite=True)
    print(report.df())  # noqa: T201
