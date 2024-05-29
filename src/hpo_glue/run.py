from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.config import Config
from hpo_glue.dataframe_utils import reduce_dtypes
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query
from hpo_glue.result import Result

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription
    from hpo_glue.problem import Problem

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


@dataclass
class Run:
    """A run of a benchmark."""

    problem: Problem
    """The problem that was run."""

    seed: int
    """The seed used for the run."""

    optimizer: type[Optimizer]
    """The optimizer to use for this problem statement"""

    optimizer_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    """The hyperparameters to use for the optimizer"""

    benchmark: BenchmarkDescription = field(init=False)
    """The benchmark that was run."""

    def __post_init__(self) -> None:
        name_parts: list[str] = [
            self.problem.name,
            f"seed={self.seed}",
            f"optimizer={self.optimizer.name}",
        ]
        if len(self.optimizer_hyperparameters) > 0:
            name_parts.append(
                ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            )
        self.name = ".".join(name_parts)
        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self.problem)
        self.benchmark = self.problem.benchmark

    class State(str, Enum):
        """The state of a problem."""

        PENDING = "PENDING"
        RUNNING = "RUNNING"
        CRASHED = "CRASHED"
        COMPLETE = "COMPLETE"

        @classmethod
        def collect(
            cls,
            state: str | Run.State | bool | Iterable[Run.State | str],
        ) -> list[Run.State]:
            """Collect state requested."""
            match state:
                case True:
                    return list(cls)
                case False:
                    return []
                case Run.State():
                    return [state]
                case str():
                    return [cls(state)]
                case _:
                    return [cls(s) if isinstance(s, str) else s for s in state]

    @dataclass
    class Report:
        """The report of a Run."""

        run: Run
        results: list[Result]

        problem: Problem = field(init=False)

        def __post_init__(self) -> None:
            self.problem = self.run.problem

        def df(  # noqa: C901, PLR0912, PLR0915
            self,
            *,
            incumbent_trajectory: bool = False,
        ) -> pd.DataFrame:
            """Return the history as a pandas DataFrame.

            Args:
                incumbent_trajectory: Whether to only include the incumbents trajectory.

            Returns:
                The history as a pandas DataFrame.
            """
            problem = self.problem

            def _encode_result(_r: Result) -> dict[str, Any]:
                _rparts: dict[str, Any] = {
                    "config.id": _r.config.config_id,
                    "query.id": _r.query.query_id,
                    "result.budget_cost": _r.budget_cost,
                    "result.budget_used_total": _r.budget_used_total,
                }
                match _r.query.fidelity:
                    case None:
                        _rparts["query.fidelity.count"] = 0
                    case (name, val):
                        _rparts["query.fidelity.count"] = 1
                        _rparts["query.fidelity.1.name"] = name
                        _rparts["query.fidelity.1.value"] = val
                    case Mapping():
                        _rparts["query.fidelity.count"] = len(_r.query.fidelity)
                        for i, (k, v) in enumerate(_r.query.fidelity.items(), start=1):
                            _rparts[f"query.fidelity.{i}.name"] = k
                            _rparts[f"query.fidelity.{i}.value"] = v

                match problem.objective:
                    case (_name, _measure):
                        _rparts["result.objective.1.value"] = _r.values[_name]
                    case Mapping():
                        for i, name in enumerate(problem.objective, start=1):
                            _rparts[f"result.cost.{i}.value"] = _r.values[name]

                match problem.fidelity:
                    case None:
                        pass
                    case (name, _):
                        assert isinstance(_r.fidelity, tuple)
                        _rparts["result.fidelity.1.value"] = _r.fidelity[1]
                    case Mapping():
                        assert isinstance(_r.fidelity, Mapping)
                        for i, name in enumerate(problem.fidelity, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.fidelity[name]

                match problem.cost:
                    case None:
                        pass
                    case (name, _):
                        _rparts["result.cost.1.value"] = _r.values[name]
                    case Mapping():
                        for i, name in enumerate(problem.cost, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.values[name]

                return _rparts

            parts = {}
            parts["run.name"] = self.run.name
            parts["problem.name"] = problem.name

            match problem.objective:
                case (name, measure):
                    parts["problem.objective.count"] = 1
                    parts["problem.objective.1.name"] = name
                    parts["problem.objective.1.minimize"] = measure.minimize
                    parts["problem.objective.1.min"] = measure.bounds[0]
                    parts["problem.objective.1.max"] = measure.bounds[1]
                case Mapping():
                    list(problem.objective)
                    parts["problem.objective.count"] = len(problem.objective)
                    for i, (k, v) in enumerate(problem.objective.items(), start=1):
                        parts[f"problem.objective.{i}.name"] = k
                        parts[f"problem.objective.{i}.minimize"] = v.minimize
                        parts[f"problem.objective.{i}.min"] = v.bounds[0]
                        parts[f"problem.objective.{i}.max"] = v.bounds[1]
                case _:
                    raise TypeError("Objective must be a tuple (name, measure) or a mapping")

            match problem.fidelity:
                case None:
                    parts["problem.fidelity.count"] = 0
                case (name, fid):
                    parts["problem.fidelity.count"] = 1
                    parts["problem.fidelity.1.name"] = name
                    parts["problem.fidelity.1.min"] = fid.min
                    parts["problem.fidelity.1.max"] = fid.max
                case Mapping():
                    list(problem.fidelity)
                    parts["problem.fidelity.count"] = len(problem.fidelity)
                    for i, (k, v) in enumerate(problem.fidelity.items(), start=1):
                        parts[f"problem.fidelity.{i}.name"] = k
                        parts[f"problem.fidelity.{i}.min"] = v.min
                        parts[f"problem.fidelity.{i}.max"] = v.max
                case _:
                    raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

            match problem.cost:
                case None:
                    parts["problem.cost.count"] = 0
                case (name, cost):
                    parts["problem.cost.count"] = 1
                    parts["problem.cost.1.name"] = name
                    parts["problem.cost.1.minimize"] = cost.minimize
                    parts["problem.cost.1.min"] = cost.bounds[0]
                    parts["problem.cost.1.max"] = cost.bounds[1]
                case Mapping():
                    list(problem.cost)
                    parts["problem.cost.count"] = len(problem.cost)
                    for i, (k, v) in enumerate(problem.cost.items(), start=1):
                        parts[f"problem.cost.{i}.name"] = k
                        parts[f"problem.cost.{i}.minimize"] = v.minimize
                        parts[f"problem.cost.{i}.min"] = v.bounds[0]
                        parts[f"problem.cost.{i}.max"] = v.bounds[1]

            _df = pd.DataFrame.from_records([_encode_result(r) for r in self.results])
            _df = _df.sort_values("result.budget_used_total", ascending=True)
            for k, v in parts.items():
                _df[k] = v

            _df["problem.benchmark"] = problem.benchmark.name
            match problem.budget:
                case TrialBudget(total):
                    _df["problem.budget.kind"] = "TrialBudget"
                    _df["problem.budget.total"] = total
                case CostBudget(total):
                    _df["problem.budget.kind"] = "CostBudget"
                    _df["problem.budget.total"] = total
                case _:
                    raise NotImplementedError(f"Unknown budget type {problem.budget}")

            _df["run.seed"] = self.run.seed
            _df["run.opt.name"] = self.run.optimizer.name

            if len(self.run.optimizer_hyperparameters) > 0:
                for k, v in self.run.optimizer_hyperparameters.items():
                    _df[f"run.opt.hp.{k}"] = v

                _df["run.opt.hp_str"] = ",".join(
                    f"{k}={v}" for k, v in self.run.optimizer_hyperparameters.items()
                )
            else:
                _df["run.opt.hp_str"] = "default"

            _df = _df.sort_values("result.budget_used_total", ascending=True)

            _df = reduce_dtypes(
                _df,
                reduce_int=True,
                reduce_float=True,
                categories=True,
                categories_exclude=("config.id", "query.id"),
            )

            if incumbent_trajectory:
                if not isinstance(self.problem.objective, tuple):
                    raise ValueError(
                        "Incumbent trajectory only supported for single objective."
                        f" Problem {self.problem.name} has {len(self.problem.objective)} objectives"
                        f" for run {self.run.name}"
                    )

                if self.problem.objective[1].minimize:
                    _df["_tmp_"] = _df["result.objective.1.value"].cummin()
                else:
                    _df["_tmp_"] = _df["result.objective.1.value"].cummax()

                _df = _df.drop_duplicates(subset="_tmp_", keep="first").drop(columns="_tmp_")  # type: ignore

            match self.problem.objective:
                case (_, measure):
                    _low, _high = measure.bounds
                    if not np.isinf(_low) and not np.isinf(_high):
                        _df["result.objective.1.normalized_value"] = (
                            _df["result.objective.1.value"] - _low
                        ) / (_high - _low)
                case Mapping():
                    for i, (_, measure) in enumerate(self.problem.objective.items(), start=1):
                        _low, _high = measure.bounds
                        if not np.isinf(_low) and not np.isinf(_high):
                            _df[f"result.objective.{i}.normalized_value"] = (
                                _df[f"result.objective.{i}.value"] - _low
                            ) / (_high - _low)
                case _:
                    raise TypeError("Objective must be a tuple (name, measure) or a mapping")

            return _df

        @classmethod
        def from_df(cls, df: pd.DataFrame, run: Run) -> Run.Report:  # noqa: C901, PLR0915
            """Load a GLUEReport from a pandas DataFrame.

            Args:
                df: The dataframe to load from. Will subselect rows
                    that match the problem name.
                run: The run definition.
            """
            problem = run.problem

            def _row_to_result(series: pd.Series) -> Result:
                _row = series.to_dict()
                _result_values: dict[str, Any] = {}
                match problem.objective:
                    case (name, _):
                        assert int(_row["problem.objective.count"]) == 1
                        assert str(_row["problem.objective.1.name"]) == name
                        _result_values[name] = _row["result.objective.1.value"]
                    case Mapping():
                        assert int(_row["problem.objective.count"]) == len(problem.objective)
                        for i, k in enumerate(problem.objective, start=1):
                            assert str(_row[f"problem.objective.{i}.name"]) == k
                            _result_values[k] = _row[f"result.objective.{i}.value"]
                    case _:
                        raise TypeError("Objective must be a tuple (name, measure) or a mapping")

                match problem.fidelity:
                    case None:
                        assert int(_row["problem.fidelity.count"]) == 0
                        _result_fidelity = None
                    case (name, fid):
                        assert int(_row["problem.fidelity.count"]) == 1
                        assert str(_row["problem.fidelity.1.name"]) == name
                        _result_fidelity = (name, fid.kind(_row["result.fidelity.1.value"]))
                    case Mapping():
                        assert int(_row["problem.fidelity.count"]) == len(problem.fidelity)
                        _result_fidelity = {}
                        for i, (name, fid) in enumerate(problem.fidelity.items(), start=1):
                            assert str(_row[f"problem.fidelity.{i}.name"]) == name
                            _result_fidelity[name] = fid.kind(_row[f"result.fidelity.{i}.value"])
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                match problem.cost:
                    case None:
                        assert int(_row["problem.cost.count"]) == 0
                    case (name, _):
                        assert int(_row["problem.cost.count"]) == 1
                        assert str(_row["problem.cost.1.name"]) == name
                        _result_values[name] = _row["result.cost.1.value"]
                    case Mapping():
                        assert int(_row["problem.cost.count"]) == len(problem.cost)
                        for i, (name, _) in enumerate(problem.cost.items(), start=1):
                            assert str(_row[f"problem.cost.{i}.name"]) == name
                            _result_values[name] = _row[f"result.cost.{i}.value"]
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                _query_f_count = int(_row["query.fidelity.count"])
                _query_fidelity: None | tuple[str, int | float] | dict[str, int | float]
                match _query_f_count:
                    case 0:
                        _query_fidelity = None
                    case 1:
                        _name = str(_row["query.fidelity.1.name"])
                        assert run.benchmark.fidelities is not None
                        _fid = run.benchmark.fidelities[_name]
                        _val = _fid.kind(_row["query.fidelity.1.value"])
                        _query_fidelity = (_name, _val)
                    case _:
                        _query_fidelity = {}
                        for i in range(1, _query_f_count + 1):
                            _name = str(_row[f"query.fidelity.{i}.name"])
                            assert run.benchmark.fidelities is not None
                            _fid = run.benchmark.fidelities[_name]
                            _query_fidelity[_name] = _fid.kind(_row[f"query.fidelity.{i}.value"])

                return Result(
                    query=Query(
                        config=Config(config_id=str(_row["config.id"]), values=None),
                        optimizer_info=None,
                        request_trajectory=False,
                        fidelity=_query_fidelity,
                    ),
                    budget_cost=float(_row["result.budget_cost"]),
                    budget_used_total=float(_row["result.budget_used_total"]),
                    values=_result_values,
                    fidelity=_result_fidelity,
                    trajectory=None,
                )

            this_run = df["run.name"] == run.name
            run_columns = [c for c in df.columns if c.startswith("run.")]
            problem_columns = [c for c in df.columns if c.startswith("problem.")]
            dup_rows = df[this_run].drop_duplicates(subset=run_columns + problem_columns)
            if len(dup_rows) > 1:
                raise ValueError(
                    f"Multiple run rows found for the provided df for run '{run.name}'"
                    f"\n{dup_rows}"
                )

            df = df[this_run].sort_values("result.budget_used_total")  # noqa: PD901
            return cls(
                run=run,
                results=[_row_to_result(row) for _, row in df[this_run].iterrows()],
            )

        def save(self, path: Path) -> None:
            """Save the report to a path."""
            self.df().to_parquet(path, index=False)

        @classmethod
        def from_path(cls, path: Path, problem: Run) -> Run.Report:
            """Load a report from a path."""
            df = pd.read_parquet(path)  # noqa: PD901
            return cls.from_df(df, run=problem)
