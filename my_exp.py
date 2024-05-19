from __future__ import annotations

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.optimizers.smac import SMAC_BO
from hpo_glue.problem import Problem

non_tabular_benchmarks = [b for b in BENCHMARKS.values() if not b.is_tabular]

problems = Problem.generate(
    optimizers=SMAC_BO,
    benchmarks=non_tabular_benchmarks,
    budget=100,
    objectives=1,
    seeds=5,
    on_error="ignore",
)
print(len(list(problems)))  # noqa: T201   # 45

problems = SMAC_BO.generate_problems(
    benchmarks=non_tabular_benchmarks,
    hyperparameters=[{"xi": 0.1}, {"xi": 0.2}, {"xi": 0.3}],
    budget=100,
    seeds=5,
    objectives=1,
    on_error="ignore",
)
print(len(list(problems)))  # noqa: T201  # 135

problems = BENCHMARKS["mfh3_good"].generate_problems(
    optimizers=SMAC_BO,
    budget=100,
    seeds=5,
    objectives=1,
    on_error="raise",
)
print(len(list(problems)))  # noqa: T201  # 1
