from __future__ import annotations

import logging
import shutil
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import pandas as pd
from tqdm import tqdm

from hpo_glue.benchmark import BenchmarkDescription
from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.optimizer import Optimizer
from hpo_glue.run import Run
from hpo_glue.utils import reduce_dtypes

if TYPE_CHECKING:
    from hpo_glue.budget import BudgetType
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure
    from hpo_glue.problem import Problem

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

T = TypeVar("T", bound=Hashable)

logger = logging.getLogger(__name__)


def _try_delete_if_exists(path: Path) -> None:
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error deleting {path}: {e}")


class RunPaths:
    """Paths for a run."""

    def __init__(self, run: Run, expdir: Path) -> None:
        """Create a new RunPaths instance."""
        self.working_dir = expdir / run.name
        self.complete_flag = self.working_dir / "complete.flag"
        self.error_file = self.working_dir / "error.txt"
        self.running_flag = self.working_dir / "running.flag"
        self.df_path = expdir / "dfs" / f"{run.name}.parquet"


@dataclass
class Experiment:
    """An experiment to run."""

    runs: list[Run]
    """The runs to run."""

    expdir: Path = field(default=DEFAULT_RELATIVE_EXP_DIR)
    """Default directory to use for experiments."""

    def run(
        self,
        *,
        on_error: Literal["raise", "continue"] = "raise",
        overwrite: Run.State | str | Sequence[Run.State | str] | bool = False,
    ) -> Experiment.Report:
        """Run the Run.

        Args:
            on_error: How to handle errors. In any case, the error will be written
                into the [`working_dir`][hpo_glue.run.Run.working_dir]
                of the problem.

                * If "raise", raise an error.
                * If "continue", log the error and continue.

            overwrite: What to overwrite.

                * If a single value, overwrites problem in that state,
                * If a list of states, overwrites any problem in one of those
                 states.
                * If `True`, overwrite problems in all states.
                * If `False`, don't overwrite any problems.
        """
        from hpo_glue._run import _run

        overwrites = Run.State.collect(overwrite)

        reports = {}
        for run in tqdm(self.runs, desc=f"Running {self.expdir}"):
            report = _run(run=run, experiment=self, on_error=on_error, overwrite=overwrites)
            reports[run.name] = report

        return Experiment.Report(experiment=self, reports=reports)

    def set_state(
        self,
        run: Run,
        state: Run.State,
        *,
        df: pd.DataFrame | None = None,
        err_tb: tuple[Exception, str] | None = None,
    ) -> None:
        """Set the run to a certain state.

        Args:
            run: The run to set the state for.
            state: The state to set the problem to.
            df: Optional dataframe to save if setting to [`Run.State.COMPLETE`].
            err_tb: Optional error traceback to save if setting to [`Run.State.CRASHED`].
        """
        paths = self.run_paths(run=run)
        match state:
            case Run.State.PENDING:
                # NOTE: Working dir takes care of flags
                for _file in (paths.working_dir, paths.df_path):
                    _try_delete_if_exists(_file)
            case Run.State.RUNNING:
                for _file in (
                    paths.df_path,
                    paths.complete_flag,
                    paths.error_file,
                ):
                    _try_delete_if_exists(_file)

                paths.working_dir.mkdir(parents=True, exist_ok=True)
                paths.df_path.parent.mkdir(parents=True, exist_ok=True)
                paths.running_flag.touch()
            case Run.State.CRASHED:
                for _file in (
                    paths.complete_flag,
                    paths.running_flag,
                    paths.df_path,
                ):
                    _try_delete_if_exists(_file)

                with paths.error_file.open("w") as f:
                    if err_tb is None:
                        f.write("None")
                    else:
                        exc, tb = err_tb
                        f.write(f"{tb}\n{exc}")

            case Run.State.COMPLETE:
                paths.complete_flag.touch()

                if df is not None:
                    df.to_parquet(paths.df_path)

                for flag in (paths.error_file, paths.running_flag):
                    _try_delete_if_exists(flag)
            case _:
                raise ValueError(f"Unknown state {state}")

    def state(self, run: Run) -> Run.State:
        """Return the state of the run.

        Args:
            run: The run to get the state for.
        """
        paths = self.run_paths(run)
        if paths.complete_flag.exists():
            return Run.State.COMPLETE

        if paths.error_file.exists():
            return Run.State.CRASHED

        if paths.running_flag.exists():
            return Run.State.RUNNING

        return Run.State.PENDING

    @classmethod
    def generate(  # noqa: PLR0913
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        budget: BudgetType | int,
        seeds: Iterable[int],
        fidelities: int = 0,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> Experiment:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            optimizers: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.
            benchmarks: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.
            expdir: Which directory to store experiment results into.
            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.
            seeds: The seed or seeds to use for the problems.
            fidelities: The number of fidelities to generate problems for.
            objectives: The number of objectives to generate problems for.
            costs: The number of costs to generate problems for.
            multi_objective_generation: The method to generate multiple objectives.
            on_error: The method to handle errors.

                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
        """
        _benchmarks: list[BenchmarkDescription] = []
        match benchmarks:
            case BenchmarkDescription():
                _benchmarks = [benchmarks]
            case Iterable():
                _benchmarks = list(benchmarks)
            case _:
                raise TypeError(
                    "Expected BenchmarkDescription or Iterable[BenchmarkDescription],"
                    f" got {type(benchmarks)}"
                )

        _problems: list[Problem] = []
        for _benchmark in _benchmarks:
            try:
                _problem = _benchmark.problem(
                    objectives=objectives,
                    budget=budget,
                    fidelities=fidelities,
                    costs=costs,
                    multi_objective_generation=multi_objective_generation,
                )
                _problems.append(_problem)
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        _experiments_per_problem: list[Experiment] = []
        for _problem in _problems:
            try:
                _exp = _problem.experiment(optimizers=optimizers, seeds=seeds, expdir=expdir)
                _experiments_per_problem.append(_exp)
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        return cls.concat(*_experiments_per_problem)

    def merge(self, *other: Experiment) -> Experiment:
        """Join two experiments together."""
        return Experiment(runs=[*self.runs, *chain.from_iterable(o.runs for o in other)])

    @classmethod
    def concat(cls, *experiments: Experiment) -> Experiment:
        """Concatenate multiple experiments together."""
        return Experiment(runs=list(chain.from_iterable(e.runs for e in experiments)))

    def groupby(self, key: Callable[[Run], T]) -> dict[T, Experiment]:
        """Group the experiments by a key."""
        groups = {}
        for problem in self.runs:
            groups.setdefault(key(problem), []).append(problem)
        return {k: Experiment(runs=problems) for k, problems in groups.items()}

    def run_paths(self, run: Run) -> RunPaths:
        """Get the paths for the run."""
        return RunPaths(run=run, expdir=self.expdir)

    @dataclass
    class Report:
        """The report for an experiment."""

        experiment: Experiment
        """The experiment that was run."""

        reports: Mapping[str, Run.Report] = field(default_factory=dict)
        """The reports."""

        def groupby(
            self,
            key: Callable[[Run.Report], T],
        ) -> list[tuple[T, Experiment.Report]]:
            """Group the experiments by a key."""
            groups: dict[T, list[Run.Report]] = {}
            for report in self.reports.values():
                groups.setdefault(key(report), []).append(report)

            return [
                (
                    k,
                    Experiment.Report(
                        experiment=Experiment(runs=[report.run for report in reports]),
                        reports={report.run.name: report for report in reports},
                    ),
                )
                for k, reports in groups.items()
            ]

        def df(self) -> pd.DataFrame:
            """Return a dictionary of dataframes of all problems."""
            dfs = [report.df() for report in self.reports.values()]
            _df = pd.concat(dfs, axis=0)
            _df = _df.convert_dtypes()
            cat_cols = [
                c
                for c in _df.select_dtypes(include=["string", object]).columns
                if c not in ("config.id", "query.id")
            ]
            _df[cat_cols] = _df[cat_cols].astype("category")
            return reduce_dtypes(_df, reduce_int=True, reduce_float=True)

        @classmethod
        def from_reports(cls, reports: list[Run.Report]) -> Experiment.Report:
            """Create an experiment from a list of reports."""
            runs = [report.run for report in reports]
            return cls(
                experiment=Experiment(runs=runs),
                reports={report.problem.name: report for report in reports},
            )

        def groupby_for_optimizer_comparison(
            self,
        ) -> list[
            tuple[
                tuple[
                    str,
                    BudgetType,
                    tuple[tuple[str, Measure], ...],
                    None | tuple[tuple[str, Fidelity], ...],
                    None | tuple[tuple[str, Measure], ...],
                ],
                Experiment.Report,
            ]
        ]:
            """Group the experiments by benchmark."""
            return list(self.groupby(lambda r: r.problem.group_for_optimizer_comparison()))
