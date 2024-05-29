from __future__ import annotations

import logging

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.experiment import Experiment
from hpo_glue.optimizers.dehb import DEHB_Optimizer
from hpo_glue.optimizers.smac import SMAC_Hyperband

logger = logging.getLogger("smac")
logger.setLevel(logging.ERROR)

# Generate single objective problems across all non-tabular benchmarks
# for SMAC_BO and Optuna with different hyperparameters
experiment = Experiment.generate(
    optimizers=[
        DEHB_Optimizer,
        SMAC_Hyperband,
        (SMAC_Hyperband, {"eta": 2}),
    ],
    benchmarks=[
        BENCHMARKS["mfh3_good"],
        BENCHMARKS["mfh6_good"],
    ],
    seeds=range(2),
    budget=50,
    objectives=1,
    fidelities=1,
    on_error="ignore",
)
experiment_report = experiment.run(overwrite=True, n_jobs=4)
for _problem, report in experiment_report.groupby_problem():
    _df = report.df(incumbent_trajectory=True)

    print(_problem.name)  # noqa: T201
    print(_df)  # noqa: T201
