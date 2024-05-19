from __future__ import annotations

import logging
import time

from hpo_glue.history import History
from hpo_glue.problem import Problem

logger = logging.getLogger(__name__)


# TODO(eddiebergman):
# 1. Do we include the results of everything in between?
#    Would be good to include it in the actual results, if wanting plot the learning
#    curve of individual configurations, not just at the gven fidelity point
# 2. Some optimizers prefer to have some kind of interuption mechanism, i.e.
# > config = opt.ask()
# > for step in steps:
# >     result = benchmark.query(config)
# >   decision = opt.tell(result)
# >   if decision == "stop":
# >         break
def run_problem(problem: Problem) -> Problem.Report:
    """Runs an optimizer on a benchmark, returning a report."""
    problem.cache_dir.mkdir(parents=True, exist_ok=True)

    benchmark = problem.benchmark.load(problem.benchmark)

    history = History()

    opt = problem.optimizer(
        problem=problem,
        working_directory=problem.cache_dir / "optimizer_dir",
        config_space=benchmark.config_space,
        seed=problem.seed,
        **problem.optimizer_hyperparameters,
    )

    budget_tracker = problem.budget.clone()

    logger.info(f"Running Problem: {problem}")
    while not budget_tracker.should_stop():
        ask_start = time.time()
        config = opt.ask()
        time.time() - ask_start

        result = benchmark.query(config)

        tell_start = time.time()
        opt.tell(result)
        time.time() - tell_start

        history.add(result)
        budget_tracker.update(result=result, problem=problem)

    return Problem.Report(problem=problem, history=history)
