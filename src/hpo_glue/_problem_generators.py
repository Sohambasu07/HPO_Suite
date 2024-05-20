from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from more_itertools import roundrobin, take

from hpo_glue.benchmark import BenchmarkDescription
from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem

if TYPE_CHECKING:
    from hpo_glue.budget import BudgetType
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure

T = TypeVar("T")

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


def _on_error(
    etype: type[Exception],
    msg: str,
    on_error: Literal["warn", "raise", "ignore"],
) -> None:
    match on_error:
        case "warn":
            _msg = msg + "\nTo ignore this warning, set `on_error='ignore'`"
            warnings.warn(_msg, stacklevel=1)
        case "raise":
            raise etype(msg)
        case "ignore":
            return


def _generate_problem_set(  # noqa: C901, PLR0911, PLR0912, PLR0915
    optimizers: (
        type[Optimizer] | OptWithHps | list[type[Optimizer]] | list[OptWithHps | type[Optimizer]]
    ),
    benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
    *,
    expdir: Path | str,
    budget: BudgetType | int,
    seeds: int | Iterable[int],
    fidelities: int = 0,
    objectives: int = 1,
    costs: int = 0,
    multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
    on_error: Literal["warn", "raise", "ignore"] = "warn",
) -> Iterator[Problem]:
    errhandler = lambda etype, msg: _on_error(etype, msg, on_error)
    optimizer: type[Optimizer]
    hps: Mapping[str, Any]
    match optimizers:
        case type():
            optimizer, hps = optimizers, {}
        case tuple():
            optimizer, hps = optimizers
        case list():
            for o in optimizers:
                yield from _generate_problem_set(
                    optimizers=o,
                    expdir=expdir,
                    benchmarks=benchmarks,
                    budget=budget,
                    seeds=seeds,
                    fidelities=fidelities,
                    objectives=objectives,
                    costs=costs,
                    multi_objective_generation=multi_objective_generation,
                    on_error=on_error,
                )
            return
        case _:
            raise ValueError("Unexpected case")

    benchmark: BenchmarkDescription
    match benchmarks:
        case BenchmarkDescription():
            benchmark = benchmarks
        case Iterable():
            for b in benchmarks:
                yield from _generate_problem_set(
                    optimizers=(optimizer, hps),
                    benchmarks=b,
                    expdir=expdir,
                    budget=budget,
                    seeds=seeds,
                    fidelities=fidelities,
                    objectives=objectives,
                    costs=costs,
                    multi_objective_generation=multi_objective_generation,
                    on_error=on_error,
                )
            return
        case _:
            raise ValueError("Unexpected case")

    match budget:
        case int():
            budget = TrialBudget(budget)
        case TrialBudget():
            budget = budget.clone()
        case CostBudget():
            raise NotImplementedError("Cost budgets are not yet supported")
        case _:
            raise TypeError(f"Unexpected type for `{budget=}`: {type(budget)}")

    # Validate fidelities request
    if fidelities < 0:
        raise ValueError(f"Number of fidelities must be greater than 0, but got `{fidelities=}`")
    if benchmark.fidelities is None:
        return _on_error(
            ValueError,
            f"Multi-fidelity is enabled but no fidelities are defined for the"
            f" benchmark {benchmark.name}",
            on_error,
        )
    if len(benchmark.fidelities) < fidelities:
        return errhandler(
            ValueError,
            f"Number of {fidelities=} requested is greater than the number of"
            f" fidelities defined for the benchmark {benchmark.name}, only"
            f" {len(benchmark.fidelities)} fidelities are defined",
        )
    if fidelities == 0 and None not in optimizer.support.fidelities:
        return errhandler(
            ValueError, f"No fidelities is not supported for optimizer {optimizer.name}"
        )
    if fidelities == 1 and "single" not in optimizer.support.fidelities:
        return errhandler(
            ValueError,
            f"Single fidelity is not supported for optimizer {optimizer.name}",
        )
    if fidelities > 1 and "many" not in optimizer.support.fidelities:
        return errhandler(
            ValueError, f"Many fidelities is not supported for optimizer {optimizer.name}"
        )

    # Validate objectives request
    if objectives < 1:
        raise ValueError(f"Number of objectives must be greater than 0, but got `{objectives=}`")

    n_costs = 0 if benchmark.costs is None else len(benchmark.costs)
    n_objectives = len(benchmark.metrics)
    if multi_objective_generation == "metric_only" and n_objectives < objectives:
        return errhandler(
            ValueError,
            f"Number of objectives requested is greater than the number of"
            f" objectives defined for the benchmark {benchmark.name}, only"
            f" {n_objectives} objectives are defined",
        )

    if multi_objective_generation == "mix_metric_cost" and n_objectives + n_costs < objectives:
        return errhandler(
            ValueError,
            f"Number of objectives requested is greater than the number of"
            f" objectives defined for the benchmark {benchmark.name}, only"
            f" {n_objectives + n_costs} objectives + costs are defined",
        )
    if objectives == 1 and "single" not in optimizer.support.objectives:
        return errhandler(
            ValueError, f"Single objective is not supported for optimizer {optimizer.name}"
        )
    if objectives > 1 and "many" not in optimizer.support.objectives:
        return errhandler(
            ValueError, f"Many objectives is not supported for optimizer {optimizer.name}"
        )

    # Validate costs request
    if costs < 0:
        raise ValueError(f"Number of costs must be greater than 0, but got `{costs=}`")
    if benchmark.costs is None:
        return errhandler(
            ValueError,
            f"Cost-awareness is enabled but no costs are defined for the"
            f" benchmark {benchmark.name}",
        )
    if len(benchmark.costs) < costs:
        return errhandler(
            ValueError,
            f"Number of costs requested is greater than the number of"
            f" costs defined for the benchmark {benchmark.name}, only"
            f" {len(benchmark.costs)} costs are defined",
        )
    if costs == 0 and None not in optimizer.support.cost_awareness:
        return errhandler(ValueError, f"No costs is not supported for optimizer {optimizer.name}")
    if costs == 1 and "single" not in optimizer.support.cost_awareness:
        return errhandler(
            ValueError, f"Single cost is not supported for optimizer {optimizer.name}"
        )
    if costs > 1 and "many" not in optimizer.support.cost_awareness:
        return errhandler(ValueError, f"Many costs is not supported for optimizer {optimizer.name}")

    def first(_d: Mapping[str, T]) -> tuple[str, T]:
        return next(iter(_d.items()))

    def first_n(n: int, _d: Mapping[str, T]) -> dict[str, T]:
        return dict(take(n, _d.items()))

    def mix_n(n: int, _d1: Mapping[str, T], _d2: Mapping[str, T]) -> dict[str, T]:
        return dict(take(n, roundrobin(_d1.items(), _d2.items())))

    # Generate problems
    _fid: tuple[str, Fidelity] | Mapping[str, Fidelity] | None
    match fidelities:
        case 0:
            _fid = None
        case 1:
            _fid = first(benchmark.fidelities)
        case _:
            _fid = first_n(fidelities, benchmark.fidelities)

    _obj: tuple[str, Measure] | dict[str, Measure] | None
    match objectives, multi_objective_generation:
        # single objective
        case 1, _:
            _obj = first(benchmark.metrics)
        case _, "metric_only":
            _obj = first_n(objectives, benchmark.metrics)
        case _, "mix_metric_cost":
            _obj = mix_n(objectives, benchmark.metrics, benchmark.costs)
        case _:
            raise RuntimeError("Unexpected case")

    _cost: tuple[str, Measure] | dict[str, Measure] | None
    match costs:
        case 0:
            _cost = None
        case 1:
            _cost = first(benchmark.costs)
        case _:
            _cost = first_n(costs, benchmark.costs)

    _seeds = [seeds] if isinstance(seeds, int) else seeds
    for seed in _seeds:
        yield Problem(
            optimizer=optimizer,
            benchmark=benchmark,
            expdir=Path(expdir),
            seed=seed,
            fidelity=_fid,
            objective=_obj,
            cost=_cost,
            optimizer_hyperparameters=hps,
            budget=budget,
        )
