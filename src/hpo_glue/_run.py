from __future__ import annotations

import logging

from hpo_glue.glu import History, Problem

logger = logging.getLogger(__name__)


def run_problem(problem: Problem) -> Problem.Report:
    """Runs an optimizer on a benchmark, returning a report."""
    problem.cache_dir.mkdir(parents=True, exist_ok=True)
    benchmark = problem.benchmark()
    opt = problem.optimizer(
        problem=problem,
        working_directory=problem.cache_dir / "optimizer_dir",
        seed=problem.seed,
        **problem.optimizer_hyperparameters,
    )

    logger.info(f"Running Problem: {problem}")

    history = History()
    used_budget = 0
    while True:
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
        config = opt.ask()
        result = benchmark.query(config)
        opt.tell(result)

        match problem.budget_type:
            case "n_trials":
                used_budget += 1
                finished = used_budget >= problem.budget
            case "time_budget":
                assert benchmark.time_budget is not None
                used_budget += result.result[benchmark.time_budget]
                finished = used_budget >= problem.budget
            case "fidelity_budget":
                match benchmark.fidelity_space:
                    case None:
                        raise ValueError(
                            "Fidelity budget specified but no fidelities in benchmark!"
                            f"\n{benchmark.name} - {benchmark.fidelity_space}",
                        )
                    case dict():
                        raise NotImplementedError("Many fidelities not yet supported")
                    case list():
                        assert isinstance(result.query.fidelity, int | float)
                        used_budget += result.query.fidelity
                    case _:
                        raise TypeError(f"type of {benchmark.fidelity_space=} not supported")

                finished = used_budget >= problem.budget
            case _:
                raise ValueError(
                    "Invalid budget type!"
                    f" '{problem.budget_type}' not in"
                    "('n_trials', 'time_budget', 'fidelity_budget')",
                )

        history.add(result)
        if finished:
            break

    return Problem.Report(problem=problem, history=history)
