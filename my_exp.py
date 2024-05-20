from __future__ import annotations

import logging

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.optimizers.smac import SMAC_Hyperband
from hpo_glue.problem import Problem

logging.basicConfig(level=logging.INFO)
non_tabular_benchmarks = [b for b in BENCHMARKS.values() if not b.is_tabular]

logger = logging.getLogger("smac")
logger.setLevel(logging.ERROR)

# Generate single objective problems across all non-tabular benchmarks
# for SMAC_BO and Optuna with different hyperparameters
problems = Problem.generate(
    optimizers=SMAC_Hyperband,
    benchmarks=BENCHMARKS["mfh3_good"],
    budget=25,
    objectives=1,
    fidelities=1,
    seeds=range(2),
    on_error="ignore",
)
for problem in problems:
    print(problem.state())  # noqa: T201
    report = problem.run(on_error="raise", overwrite=False)
    print(report.df())  # noqa: T201
